#include "neural_model.hpp"

#include <algorithm>
#include <cmath>

namespace {

float clamp01(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

float smooth(float current, float target, float rate, float dt) {
    const float t = 1.0f - std::exp(-rate * std::max(dt, 0.0f));
    return current + (target - current) * t;
}

float satMix(float a, float b, float gain = 1.0f) {
    return clamp01((0.55f * a + 0.45f * b) * gain);
}

} // namespace

void FlyVisualCircuit::reset() {
    state_ = NeuralState{};
}

MotorCommand FlyVisualCircuit::update(const VisionFeatures& input, float dt) {
    // Topology is literature-grounded; these compact scalar dynamics and gains are
    // engineering proxies for a feasibility spike, not a biophysical simulation.
    const float luminanceDrive = clamp01(std::max(input.on, input.off) * 0.75f + input.confidence * 0.25f);
    state_.r1r6 = smooth(state_.r1r6, luminanceDrive, 18.0f, dt);

    // R1-R6 -> lamina ON/OFF channels.
    state_.l1 = smooth(state_.l1, clamp01(state_.r1r6 * (0.25f + 0.95f * input.on)), 14.0f, dt);
    state_.l2 = smooth(state_.l2, clamp01(state_.r1r6 * (0.25f + 0.95f * input.off)), 14.0f, dt);

    // ON motion pathway upstream of T4: Mi1/Tm3 with Mi4/Mi9 contextual inputs.
    state_.mi1 = smooth(state_.mi1, state_.l1, 11.0f, dt);
    state_.tm3 = smooth(state_.tm3, clamp01(0.75f * state_.l1 + 0.25f * input.on), 15.0f, dt);
    state_.mi4 = smooth(state_.mi4, clamp01(state_.l1 * 0.62f), 6.0f, dt);
    state_.mi9 = smooth(state_.mi9, clamp01(state_.l1 * 0.55f), 4.5f, dt);

    // OFF motion pathway upstream of T5.
    state_.tm1 = smooth(state_.tm1, state_.l2, 10.0f, dt);
    state_.tm2 = smooth(state_.tm2, clamp01(0.82f * state_.l2 + 0.18f * input.off), 12.0f, dt);
    state_.tm4 = smooth(state_.tm4, clamp01(state_.l2 * 0.68f), 7.0f, dt);
    state_.tm9 = smooth(state_.tm9, clamp01(state_.l2 * 0.72f), 5.0f, dt);

    const float onDrive = clamp01((state_.mi1 + state_.tm3 + 0.35f * state_.mi9) / 2.35f);
    const float offDrive = clamp01((state_.tm1 + state_.tm2 + state_.tm4 + 0.45f * state_.tm9) / 3.45f);

    const std::array<float, 4> direction = {
        input.motionLeft, input.motionRight, input.motionUp, input.motionDown
    };

    for (int i = 0; i < 4; ++i) {
        state_.t4[i] = smooth(state_.t4[i], clamp01(direction[i] * (0.20f + onDrive)), 13.0f, dt);
        state_.t5[i] = smooth(state_.t5[i], clamp01(direction[i] * (0.20f + offDrive)), 13.0f, dt);
    }

    // Lobula-plate large-field readout proxies.
    const float horizontal = clamp01(
        std::max(state_.t4[0], state_.t4[1]) + std::max(state_.t5[0], state_.t5[1]));
    const float vertical = clamp01(
        std::max(state_.t4[2], state_.t4[3]) + std::max(state_.t5[2], state_.t5[3]));
    state_.hs = smooth(state_.hs, horizontal, 7.0f, dt);
    state_.vs = smooth(state_.vs, vertical, 7.0f, dt);

    // Looming-sensitive visual projection channels. LC4 is associated with looming
    // velocity/location signals; LPLC2 contributes looming angular-size information
    // to the giant-fibre escape pathway.
    state_.lc4 = smooth(state_.lc4, clamp01(input.looming * (0.35f + 0.65f * offDrive)), 12.0f, dt);
    state_.lplc2 = smooth(state_.lplc2, clamp01(input.looming * input.looming * 1.25f), 9.0f, dt);
    state_.giantFiber = smooth(
        state_.giantFiber,
        clamp01(0.58f * state_.lc4 + 0.62f * state_.lplc2),
        18.0f,
        dt);

    const float leftDrive = satMix(state_.t4[0], state_.t5[0]);
    const float rightDrive = satMix(state_.t4[1], state_.t5[1]);
    const float upDrive = satMix(state_.t4[2], state_.t5[2]);
    const float downDrive = satMix(state_.t4[3], state_.t5[3]);

    MotorCommand motor;
    motor.escape = state_.giantFiber;

    // Motor readout is explicitly a proxy: the feasibility question is whether a
    // mapped visual circuit can close the realtime loop, not whether this reproduces
    // the complete descending/motor connectome.
    motor.yaw = std::clamp((rightDrive - leftDrive) * 0.95f
                           - input.objectX * motor.escape * 1.55f, -1.0f, 1.0f);
    motor.pitch = std::clamp((upDrive - downDrive) * 0.65f
                             - input.objectY * motor.escape * 0.55f, -1.0f, 1.0f);
    motor.forward = std::clamp(0.10f + 0.38f * (1.0f - motor.escape)
                               + 0.42f * motor.escape, 0.0f, 1.0f);
    return motor;
}
