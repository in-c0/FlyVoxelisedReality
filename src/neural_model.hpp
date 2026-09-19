#pragma once

#include <array>

struct VisionFeatures {
    float on = 0.0f;
    float off = 0.0f;
    float motionLeft = 0.0f;
    float motionRight = 0.0f;
    float motionUp = 0.0f;
    float motionDown = 0.0f;
    float looming = 0.0f;
    float objectX = 0.0f;   // -1 left .. +1 right
    float objectY = 0.0f;   // -1 down .. +1 up
    float confidence = 0.0f;
};

struct NeuralState {
    float r1r6 = 0.0f;
    float l1 = 0.0f;
    float l2 = 0.0f;

    float mi1 = 0.0f;
    float tm3 = 0.0f;
    float mi4 = 0.0f;
    float mi9 = 0.0f;

    float tm1 = 0.0f;
    float tm2 = 0.0f;
    float tm4 = 0.0f;
    float tm9 = 0.0f;

    // Aggregated directional populations, not per-column reconstructions.
    // Order: left, right, up, down.
    std::array<float, 4> t4{};
    std::array<float, 4> t5{};

    float hs = 0.0f;
    float vs = 0.0f;

    float lc4 = 0.0f;
    float lplc2 = 0.0f;
    float giantFiber = 0.0f;
};

struct MotorCommand {
    float yaw = 0.0f;       // + right
    float pitch = 0.0f;     // + up
    float forward = 0.0f;
    float escape = 0.0f;
};

class FlyVisualCircuit {
public:
    void reset();
    MotorCommand update(const VisionFeatures& input, float dt);

    const NeuralState& state() const { return state_; }

private:
    NeuralState state_{};
};
