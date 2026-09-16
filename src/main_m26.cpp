#include "depth_estimator.hpp"
#include "preset_scenes.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kGridW = 48;
constexpr int kGridH = 36;
constexpr int kDepthInferenceInterval = 2;
constexpr int kMaxInstances = 20000;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kVirtualCameraFovDegrees = 68.0f;
constexpr float kNearDistance = 0.75f;
constexpr float kFarDistance = 4.50f;
constexpr float kVoxelQuantisation = 0.075f;

struct Instance {
    glm::vec4 positionScale;
    glm::vec4 colorMotion;
};

struct AppState {
    int framebufferW = 1280;
    int framebufferH = 720;
    int sceneMode = 1; // 1 office, 2 kitchen, 3 live CV
    bool paused = false;
    bool dragging = false;
    bool depthEnabled = true;
    bool agentEnabled = true;
    bool enhancedLighting = true;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    float yaw = 0.0f;
    float pitch = 0.05f;
    float distance = 8.6f;
};

struct VisualStimulus {
    bool valid = false;
    float activity = 0.0f;
    float leftMotion = 0.0f;
    float rightMotion = 0.0f;
    glm::vec3 worldTarget{0.0f};
};

struct FlyAgent {
    glm::vec3 position{0.0f, 0.0f, 2.35f};
    glm::vec3 forward{0.0f, 0.0f, -1.0f};
    float retina = 0.0f;
    float lamina = 0.0f;
    float medulla = 0.0f;
    float lobula = 0.0f;
};

float quantise(float value, float step) {
    return std::round(value / step) * step;
}

float smoothToward(float current, float target, float rate, float dt) {
    const float t = 1.0f - std::exp(-rate * std::max(dt, 0.0f));
    return current + (target - current) * t;
}

std::string resolveDepthModelPath(int argc, char** argv) {
    if (argc > 1 && argv[1] != nullptr) return argv[1];

    const std::vector<std::filesystem::path> candidates = {
        "models/model-small.onnx",
        "../../../models/model-small.onnx",
        "../../models/model-small.onnx"
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) return candidate.string();
    }
    return candidates.front().string();
}

glm::vec3 backProjectGridPoint(float x, float y, float nearValue) {
    const float halfW = static_cast<float>(kGridW - 1) * 0.5f;
    const float halfH = static_cast<float>(kGridH - 1) * 0.5f;
    const float fovRadians = kVirtualCameraFovDegrees * kPi / 180.0f;
    const float focalPixels = (0.5f * static_cast<float>(kGridW)) / std::tan(fovRadians * 0.5f);
    const float depthMidpoint = (kNearDistance + kFarDistance) * 0.5f;
    const float distance = kFarDistance - std::clamp(nearValue, 0.0f, 1.0f) * (kFarDistance - kNearDistance);

    return {
        ((x - halfW) / focalPixels) * distance,
        ((halfH - y) / focalPixels) * distance,
        -(distance - depthMidpoint)
    };
}

GLuint compileShader(GLenum type, const char* source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok != GL_TRUE) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<size_t>(std::max(length, 1)), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error("Shader compilation failed:\n" + log);
    }
    return shader;
}

GLuint createProgram() {
    static constexpr const char* kVertexShader = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 iPositionScale;
layout(location = 3) in vec4 iColorMotion;

uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vWorldPosition;
out vec3 vNormal;
out vec3 vColor;
out float vMotion;

void main() {
    vec3 world = iPositionScale.xyz + aPosition * iPositionScale.w;
    vWorldPosition = world;
    vNormal = aNormal;
    vColor = iColorMotion.rgb;
    vMotion = iColorMotion.a;
    gl_Position = uProjection * uView * vec4(world, 1.0);
}
)GLSL";

    static constexpr const char* kFragmentShader = R"GLSL(
#version 330 core
in vec3 vWorldPosition;
in vec3 vNormal;
in vec3 vColor;
in float vMotion;

uniform vec3 uCameraPosition;
uniform int uSceneMode;
uniform int uEnhancedLighting;

out vec4 FragColor;

vec3 acesApprox(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCameraPosition - vWorldPosition);
    float activity = clamp(vMotion, 0.0, 1.0);
    vec3 activityColour = vec3(0.10, 1.00, 0.72);
    vec3 base = mix(vColor, activityColour, activity * 0.48);

    if (uEnhancedLighting == 0) {
        vec3 L = normalize(vec3(-0.45, 0.75, 0.55));
        float diffuse = max(dot(N, L), 0.0);
        float rim = pow(1.0 - abs(N.z), 3.0) * 0.08;
        vec3 lit = base * (0.30 + 0.70 * diffuse) + activityColour * rim * activity;
        FragColor = vec4(lit, 1.0);
        return;
    }

    // Warm high sun plus a cool, broad window/sky source from the back-right.
    vec3 sunDir = normalize(vec3(-0.38, 0.82, 0.42));
    vec3 windowDir = normalize(vec3(0.72, 0.38, -0.78));
    float sunNdotL = max(dot(N, sunDir), 0.0);
    float windowNdotL = max(dot(N, windowDir), 0.0);

    vec3 sunColour = vec3(1.00, 0.78, 0.55);
    vec3 skyColour = vec3(0.46, 0.62, 0.82);
    vec3 groundColour = vec3(0.22, 0.16, 0.11);

    // Hemisphere ambient keeps wall-facing surfaces readable while retaining form.
    float hemi = N.y * 0.5 + 0.5;
    vec3 ambient = mix(groundColour, skyColour, hemi) * 0.52;

    // The preset rooms place their main window on the right/back wall. This broad
    // term is intentionally soft rather than a hard painted light patch.
    float presetMask = uSceneMode == 3 ? 0.25 : 1.0;
    float windowProximity = 1.0;
    if (uSceneMode != 3) {
        float dx = max(0.0, 2.7 - vWorldPosition.x);
        float dz = abs(vWorldPosition.z + 2.4);
        windowProximity = 0.55 + 0.45 * exp(-0.22 * (dx * dx + dz * dz));
    }
    vec3 windowLight = skyColour * windowNdotL * (0.42 * presetMask) * windowProximity;
    vec3 direct = sunColour * sunNdotL * 1.10 + windowLight;

    // Small material heuristic: blue window voxels and dark manufactured surfaces
    // receive tighter highlights; wood/plaster remain rougher.
    float luminance = dot(base, vec3(0.2126, 0.7152, 0.0722));
    float blueBias = max(base.b - max(base.r, base.g), 0.0);
    float glassLike = smoothstep(0.08, 0.30, blueBias) * smoothstep(0.40, 0.75, base.b);
    float darkManufactured = 1.0 - smoothstep(0.10, 0.30, luminance);
    float specularStrength = 0.05 + glassLike * 0.52 + darkManufactured * 0.18;
    float shininess = mix(20.0, 92.0, max(glassLike, darkManufactured * 0.55));
    vec3 H = normalize(sunDir + V);
    float specular = pow(max(dot(N, H), 0.0), shininess) * specularStrength * sunNdotL;

    // Cheap world-space contact darkening: vertical faces close to the floor become
    // slightly darker, which grounds chair/table legs without baking dark floor tiles.
    float verticalFace = 1.0 - abs(N.y);
    float floorDistance = abs(vWorldPosition.y + 1.55);
    float contact = exp(-floorDistance * 5.5) * verticalFace;
    float contactAO = 1.0 - 0.34 * contact;

    // Additional cavity-like shading on downward-facing surfaces.
    float downFacing = max(-N.y, 0.0);
    float ambientOcclusion = clamp(contactAO * (1.0 - 0.13 * downFacing), 0.58, 1.0);

    vec3 colour = base * (ambient * ambientOcclusion + direct) + sunColour * specular;

    // Preserve live neural activity as a restrained emissive cue.
    colour += activityColour * activity * 0.28;

    // Gentle depth haze separates foreground/midground/background without hiding voxels.
    float cameraDistance = length(uCameraPosition - vWorldPosition);
    float fog = 1.0 - exp(-0.0024 * cameraDistance * cameraDistance);
    vec3 fogColour = uSceneMode == 3 ? vec3(0.035, 0.050, 0.065) : vec3(0.10, 0.115, 0.12);
    colour = mix(colour, fogColour, clamp(fog, 0.0, 0.32));

    // Exposure + filmic tone mapping + display gamma.
    colour *= 1.08;
    colour = acesApprox(colour);
    colour = pow(colour, vec3(1.0 / 2.2));

    FragColor = vec4(colour, 1.0);
}
)GLSL";

    const GLuint vs = compileShader(GL_VERTEX_SHADER, kVertexShader);
    const GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    const GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<size_t>(std::max(length, 1)), '\0');
        glGetProgramInfoLog(program, length, nullptr, log.data());
        glDeleteProgram(program);
        throw std::runtime_error("Program link failed:\n" + log);
    }
    return program;
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (state) {
        state->framebufferW = std::max(width, 1);
        state->framebufferH = std::max(height, 1);
    }
    glViewport(0, 0, width, height);
}

void keyCallback(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (!state) return;

    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_SPACE) state->paused = !state->paused;
    if (key == GLFW_KEY_D) state->depthEnabled = !state->depthEnabled;
    if (key == GLFW_KEY_A) state->agentEnabled = !state->agentEnabled;
    if (key == GLFW_KEY_L) state->enhancedLighting = !state->enhancedLighting;
    if (key == GLFW_KEY_1) state->sceneMode = 1;
    if (key == GLFW_KEY_2) state->sceneMode = 2;
    if (key == GLFW_KEY_3) state->sceneMode = 3;
    if (key == GLFW_KEY_R) {
        state->yaw = 0.0f;
        state->pitch = 0.05f;
        state->distance = state->sceneMode == 3 ? 7.2f : 8.6f;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (!state) return;

    if (action == GLFW_PRESS) {
        state->dragging = true;
        glfwGetCursorPos(window, &state->lastMouseX, &state->lastMouseY);
    } else if (action == GLFW_RELEASE) {
        state->dragging = false;
    }
}

void cursorPositionCallback(GLFWwindow* window, double x, double y) {
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (!state || !state->dragging) return;

    const double dx = x - state->lastMouseX;
    const double dy = y - state->lastMouseY;
    state->lastMouseX = x;
    state->lastMouseY = y;

    state->yaw -= static_cast<float>(dx) * 0.006f;
    state->pitch -= static_cast<float>(dy) * 0.006f;
    state->pitch = std::clamp(state->pitch, -1.25f, 1.25f);
}

void scrollCallback(GLFWwindow* window, double, double yoffset) {
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (!state) return;
    state->distance = std::clamp(state->distance - static_cast<float>(yoffset) * 0.45f, 2.4f, 18.0f);
}

cv::Mat makeSyntheticFrame(double t) {
    cv::Mat frame(480, 640, CV_8UC3);
    for (int y = 0; y < frame.rows; ++y) {
        auto* row = frame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < frame.cols; ++x) {
            row[x] = cv::Vec3b(
                static_cast<unsigned char>((x * 255) / frame.cols),
                static_cast<unsigned char>((y * 255) / frame.rows),
                static_cast<unsigned char>(70));
        }
    }

    const int cx = static_cast<int>(frame.cols * (0.5 + 0.30 * std::sin(t * 1.3)));
    const int cy = static_cast<int>(frame.rows * (0.5 + 0.22 * std::cos(t * 1.7)));
    cv::circle(frame, {cx, cy}, 65, cv::Scalar(230, 230, 255), cv::FILLED, cv::LINE_AA);
    return frame;
}

cv::Mat luminanceFallbackNearness(const cv::Mat& bgrSmall) {
    cv::Mat gray;
    cv::cvtColor(bgrSmall, gray, cv::COLOR_BGR2GRAY);
    cv::Mat nearness;
    gray.convertTo(nearness, CV_32F, 1.0 / 255.0);
    return nearness;
}

std::vector<Instance> frameToVoxels(
    const cv::Mat& bgrSmall,
    const cv::Mat& flow,
    const cv::Mat& nearness) {

    std::vector<Instance> instances;
    instances.reserve(static_cast<size_t>(kGridW * kGridH + 16));

    for (int y = 0; y < kGridH; ++y) {
        const auto* pixels = bgrSmall.ptr<cv::Vec3b>(y);
        const cv::Point2f* flowRow = flow.empty() ? nullptr : flow.ptr<cv::Point2f>(y);
        const float* depthRow = nearness.empty() ? nullptr : nearness.ptr<float>(y);

        for (int x = 0; x < kGridW; ++x) {
            const cv::Vec3b bgr = pixels[x];
            const glm::vec3 colour(
                static_cast<float>(bgr[2]) / 255.0f,
                static_cast<float>(bgr[1]) / 255.0f,
                static_cast<float>(bgr[0]) / 255.0f);

            float motion = 0.0f;
            if (flowRow) {
                const cv::Point2f f = flowRow[x];
                motion = std::clamp(std::sqrt(f.x * f.x + f.y * f.y) * 0.22f, 0.0f, 1.0f);
            }

            const float nearValue = depthRow ? depthRow[x] : 0.5f;
            glm::vec3 world = backProjectGridPoint(static_cast<float>(x), static_cast<float>(y), nearValue);
            world.x = quantise(world.x, kVoxelQuantisation);
            world.y = quantise(world.y, kVoxelQuantisation);
            world.z = quantise(world.z, kVoxelQuantisation);

            const float distance = kFarDistance - std::clamp(nearValue, 0.0f, 1.0f) * (kFarDistance - kNearDistance);
            const float distance01 = (distance - kNearDistance) / (kFarDistance - kNearDistance);
            const float scale = 0.045f + distance01 * 0.030f + motion * 0.015f;
            instances.push_back({glm::vec4(world, scale), glm::vec4(colour, motion)});
        }
    }

    return instances;
}

std::vector<Instance> presetToInstances(const PresetScene& scene) {
    std::vector<Instance> instances;
    instances.reserve(scene.voxels.size() + 16);
    for (const SceneVoxel& voxel : scene.voxels) {
        instances.push_back({glm::vec4(voxel.position, voxel.scale), glm::vec4(voxel.colour, 0.0f)});
    }
    return instances;
}

VisualStimulus extractVisualStimulus(const cv::Mat& flow, const cv::Mat& nearness) {
    VisualStimulus stimulus;
    if (flow.empty()) return stimulus;

    double totalWeight = 0.0;
    double weightedX = 0.0;
    double weightedY = 0.0;
    double left = 0.0;
    double right = 0.0;

    for (int y = 0; y < flow.rows; ++y) {
        const cv::Point2f* row = flow.ptr<cv::Point2f>(y);
        for (int x = 0; x < flow.cols; ++x) {
            const cv::Point2f f = row[x];
            const float magnitude = std::sqrt(f.x * f.x + f.y * f.y);
            const float weight = std::max(magnitude - 0.12f, 0.0f);
            if (weight <= 0.0f) continue;

            totalWeight += weight;
            weightedX += static_cast<double>(x) * weight;
            weightedY += static_cast<double>(y) * weight;
            if (x < flow.cols / 2) left += weight;
            else right += weight;
        }
    }

    const double normaliser = static_cast<double>(flow.rows * flow.cols) * 0.55;
    stimulus.activity = static_cast<float>(std::clamp(totalWeight / normaliser, 0.0, 1.0));
    stimulus.leftMotion = static_cast<float>(std::clamp(left / (normaliser * 0.5), 0.0, 1.0));
    stimulus.rightMotion = static_cast<float>(std::clamp(right / (normaliser * 0.5), 0.0, 1.0));

    if (totalWeight < 2.0) return stimulus;

    const float cx = static_cast<float>(weightedX / totalWeight);
    const float cy = static_cast<float>(weightedY / totalWeight);
    const int ix = std::clamp(static_cast<int>(std::round(cx)), 0, kGridW - 1);
    const int iy = std::clamp(static_cast<int>(std::round(cy)), 0, kGridH - 1);
    const float nearValue = nearness.empty() ? 0.5f : nearness.at<float>(iy, ix);

    stimulus.valid = true;
    stimulus.worldTarget = backProjectGridPoint(cx, cy, nearValue);
    return stimulus;
}

VisualStimulus makePresetStimulus(const PresetScene& scene, double elapsed, int sceneMode) {
    VisualStimulus stimulus;
    const float t = static_cast<float>(elapsed);
    const float phase = sceneMode == 1 ? 0.0f : 1.2f;
    const float lateral = std::sin(t * 0.75f + phase) * 0.85f;
    const float vertical = std::sin(t * 1.15f + phase) * 0.20f;

    stimulus.valid = true;
    stimulus.activity = 0.38f + 0.22f * (0.5f + 0.5f * std::sin(t * 1.7f));
    stimulus.leftMotion = lateral < 0.0f ? stimulus.activity : 0.0f;
    stimulus.rightMotion = lateral >= 0.0f ? stimulus.activity : 0.0f;
    stimulus.worldTarget = scene.stimulusAnchor + glm::vec3(lateral, vertical, 0.35f * std::cos(t * 0.55f));
    return stimulus;
}

void updateFlyAgent(FlyAgent& fly, const VisualStimulus& stimulus, float dt, bool enabled) {
    fly.retina = smoothToward(fly.retina, stimulus.activity, 12.0f, dt);
    fly.lamina = smoothToward(fly.lamina, fly.retina, 8.0f, dt);
    fly.medulla = smoothToward(fly.medulla, fly.lamina, 5.0f, dt);
    fly.lobula = smoothToward(fly.lobula, fly.medulla, 3.5f, dt);

    if (!enabled || !stimulus.valid || stimulus.activity < 0.02f) return;

    glm::vec3 delta = stimulus.worldTarget - fly.position;
    const float distance = glm::length(delta);
    if (distance < 0.001f) return;

    const glm::vec3 desiredDirection = delta / distance;
    const float turnT = std::clamp(dt * (1.5f + 4.0f * fly.lobula), 0.0f, 1.0f);
    fly.forward = glm::normalize(fly.forward * (1.0f - turnT) + desiredDirection * turnT);

    // Engineering proxy only. M3 replaces this with connectivity grounded in a
    // real Drosophila visual-neural map rather than claiming biological fidelity.
    const float speed = 0.10f + 0.85f * fly.lobula;
    fly.position += fly.forward * std::min(speed * dt, distance * 0.18f);
    fly.position.x = std::clamp(fly.position.x, -4.0f, 4.0f);
    fly.position.y = std::clamp(fly.position.y, -3.0f, 3.0f);
    fly.position.z = std::clamp(fly.position.z, -2.8f, 3.0f);
}

void appendFlyInstances(std::vector<Instance>& instances, const FlyAgent& fly) {
    glm::vec3 forward = fly.forward;
    if (glm::length(forward) < 0.001f) forward = {0.0f, 0.0f, -1.0f};
    forward = glm::normalize(forward);

    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::cross(forward, up);
    if (glm::length(right) < 0.001f) right = {1.0f, 0.0f, 0.0f};
    right = glm::normalize(right);

    const glm::vec3 bodyColour(0.55f, 0.38f, 0.14f);
    const glm::vec3 headColour(0.25f, 0.12f, 0.08f);
    const glm::vec3 wingColour(0.58f, 0.76f, 0.88f);

    instances.push_back({glm::vec4(fly.position, 0.12f), glm::vec4(bodyColour, fly.lobula)});
    instances.push_back({glm::vec4(fly.position + forward * 0.17f, 0.085f), glm::vec4(headColour, fly.retina)});
    instances.push_back({glm::vec4(fly.position - right * 0.15f + up * 0.05f, 0.075f), glm::vec4(wingColour, fly.medulla)});
    instances.push_back({glm::vec4(fly.position + right * 0.15f + up * 0.05f, 0.075f), glm::vec4(wingColour, fly.medulla)});

    const float stages[4] = {fly.retina, fly.lamina, fly.medulla, fly.lobula};
    for (int i = 0; i < 4; ++i) {
        const glm::vec3 p = fly.position + glm::vec3(-0.27f + static_cast<float>(i) * 0.18f, 0.30f, 0.0f);
        const glm::vec3 c(0.10f + 0.10f * i, 0.28f + 0.10f * i, 0.45f + 0.08f * i);
        instances.push_back({glm::vec4(p, 0.055f), glm::vec4(c, stages[i])});
    }
}

void appendStimulusProbe(std::vector<Instance>& instances, const VisualStimulus& stimulus) {
    if (!stimulus.valid) return;
    const glm::vec3 colour(0.95f, 0.48f, 0.08f);
    instances.push_back({glm::vec4(stimulus.worldTarget, 0.09f), glm::vec4(colour, stimulus.activity)});
}

void resetFlyForMode(FlyAgent& fly, int sceneMode, const PresetScene& office, const PresetScene& kitchen) {
    fly = FlyAgent{};
    if (sceneMode == 1) fly.position = office.flyStart;
    else if (sceneMode == 2) fly.position = kitchen.flyStart;
}

const char* sceneModeName(int sceneMode) {
    if (sceneMode == 1) return "OFFICE";
    if (sceneMode == 2) return "KITCHEN";
    return "LIVE CV";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (!glfwInit()) throw std::runtime_error("GLFW initialisation failed");

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        GLFWwindow* window = glfwCreateWindow(1280, 720, "FlyVoxelisedReality | M2.6 LIGHTING", nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("Could not create GLFW window");
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        glewExperimental = GL_TRUE;
        const GLenum glewResult = glewInit();
        if (glewResult != GLEW_OK) {
            glfwDestroyWindow(window);
            glfwTerminate();
            throw std::runtime_error("GLEW initialisation failed: " + std::string(reinterpret_cast<const char*>(glGetString(GL_VERSION))));
        }
        glGetError();

        AppState state;
        glfwSetWindowUserPointer(window, &state);
        glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPositionCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwGetFramebufferSize(window, &state.framebufferW, &state.framebufferH);
        glViewport(0, 0, state.framebufferW, state.framebufferH);

        std::cout << "OpenGL: " << glGetString(GL_VERSION) << '\n';
        std::cout << "GPU:    " << glGetString(GL_RENDERER) << '\n';

        const PresetScene office = makeOfficeScene();
        const PresetScene kitchen = makeKitchenScene();
        std::cout << "Scenes: 1 Office / Studio (" << office.voxels.size() << " voxels)\n";
        std::cout << "        2 Kitchen / Fruit Table (" << kitchen.voxels.size() << " voxels)\n";
        std::cout << "        3 Live Camera Reconstruction\n";

        const std::string depthModelPath = resolveDepthModelPath(argc, argv);
        DepthEstimator depthEstimator(depthModelPath);
        std::cout << "Depth:  " << depthEstimator.status() << '\n';
        if (!depthEstimator.available()) std::cout << "        Run scripts/download_depth_model.ps1, then restart for live depth.\n";
        std::cout << "M2.6:   world-space lighting + window fill + contact shading + tone mapping\n";
        std::cout << "Keys:   1 office | 2 kitchen | 3 live CV | L lighting | A agent | D depth | Space pause | R reset\n";

        const GLuint program = createProgram();
        const GLint locView = glGetUniformLocation(program, "uView");
        const GLint locProjection = glGetUniformLocation(program, "uProjection");
        const GLint locCameraPosition = glGetUniformLocation(program, "uCameraPosition");
        const GLint locSceneMode = glGetUniformLocation(program, "uSceneMode");
        const GLint locEnhancedLighting = glGetUniformLocation(program, "uEnhancedLighting");

        static constexpr float cubeVertices[] = {
            -1,-1,-1,  0,0,-1,   1,1,-1,  0,0,-1,   1,-1,-1,  0,0,-1,
             1,1,-1,   0,0,-1,  -1,-1,-1, 0,0,-1,  -1,1,-1,   0,0,-1,
            -1,-1,1,   0,0,1,    1,-1,1,  0,0,1,    1,1,1,    0,0,1,
             1,1,1,    0,0,1,   -1,1,1,   0,0,1,   -1,-1,1,   0,0,1,
            -1,1,1,   -1,0,0,   -1,1,-1, -1,0,0,   -1,-1,-1, -1,0,0,
            -1,-1,-1, -1,0,0,   -1,-1,1, -1,0,0,   -1,1,1,   -1,0,0,
             1,1,1,    1,0,0,    1,-1,-1, 1,0,0,    1,1,-1,    1,0,0,
             1,-1,-1,  1,0,0,    1,1,1,   1,0,0,    1,-1,1,    1,0,0,
            -1,-1,-1,  0,-1,0,   1,-1,-1, 0,-1,0,   1,-1,1,    0,-1,0,
             1,-1,1,   0,-1,0,  -1,-1,1,  0,-1,0,  -1,-1,-1,  0,-1,0,
            -1,1,-1,   0,1,0,    1,1,1,   0,1,0,    1,1,-1,    0,1,0,
             1,1,1,    0,1,0,   -1,1,-1,  0,1,0,   -1,1,1,     0,1,0
        };

        GLuint vao = 0;
        GLuint cubeVbo = 0;
        GLuint instanceVbo = 0;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &cubeVbo);
        glGenBuffers(1, &instanceVbo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, cubeVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));

        glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
        glBufferData(GL_ARRAY_BUFFER, kMaxInstances * sizeof(Instance), nullptr, GL_STREAM_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), nullptr);
        glVertexAttribDivisor(2, 1);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), reinterpret_cast<void*>(sizeof(glm::vec4)));
        glVertexAttribDivisor(3, 1);
        glBindVertexArray(0);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glClearColor(0.018f, 0.024f, 0.032f, 1.0f);

        cv::VideoCapture capture(0, cv::CAP_ANY);
        if (capture.isOpened()) {
            capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            capture.set(cv::CAP_PROP_FPS, 30);
            std::cout << "Camera: live webcam input ready for mode 3\n";
        } else {
            std::cerr << "Camera unavailable; mode 3 will use synthetic fallback input.\n";
        }

        cv::Mat frame;
        cv::Mat small;
        cv::Mat gray;
        cv::Mat previousGray;
        cv::Mat flow;
        cv::Mat depthNearness;
        std::vector<Instance> instances;
        FlyAgent fly;
        VisualStimulus stimulus;
        resetFlyForMode(fly, state.sceneMode, office, kitchen);

        int activeSceneMode = state.sceneMode;
        int capturedFrameIndex = 0;
        double titleAccumulator = 0.0;
        int titleFrames = 0;
        auto previousTime = std::chrono::steady_clock::now();
        const auto startTime = previousTime;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            const auto now = std::chrono::steady_clock::now();
            const float dt = std::chrono::duration<float>(now - previousTime).count();
            previousTime = now;
            const double elapsed = std::chrono::duration<double>(now - startTime).count();

            if (activeSceneMode != state.sceneMode) {
                activeSceneMode = state.sceneMode;
                resetFlyForMode(fly, activeSceneMode, office, kitchen);
                stimulus = VisualStimulus{};
                previousGray.release();
                flow.release();
                state.distance = activeSceneMode == 3 ? 7.2f : 8.6f;
            }

            if (!state.paused) {
                if (state.sceneMode == 3) {
                    bool gotFrame = false;
                    if (capture.isOpened()) gotFrame = capture.read(frame);
                    if (!gotFrame) frame = makeSyntheticFrame(elapsed);
                    else cv::flip(frame, frame, 1);

                    cv::resize(frame, small, cv::Size(kGridW, kGridH), 0.0, 0.0, cv::INTER_AREA);
                    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
                    if (!previousGray.empty()) cv::calcOpticalFlowFarneback(previousGray, gray, flow, 0.5, 3, 9, 2, 5, 1.1, 0);
                    else flow = cv::Mat::zeros(gray.size(), CV_32FC2);
                    gray.copyTo(previousGray);

                    if (state.depthEnabled && depthEstimator.available()) {
                        if (depthNearness.empty() || (capturedFrameIndex % kDepthInferenceInterval) == 0) {
                            cv::Mat inferred = depthEstimator.inferNearness(frame, cv::Size(kGridW, kGridH));
                            if (!inferred.empty()) depthNearness = inferred;
                        }
                    } else {
                        depthNearness = luminanceFallbackNearness(small);
                    }
                    if (depthNearness.empty()) depthNearness = luminanceFallbackNearness(small);

                    stimulus = extractVisualStimulus(flow, depthNearness);
                    instances = frameToVoxels(small, flow, depthNearness);
                    ++capturedFrameIndex;
                } else {
                    const PresetScene& scene = state.sceneMode == 1 ? office : kitchen;
                    stimulus = makePresetStimulus(scene, elapsed, state.sceneMode);
                    instances = presetToInstances(scene);
                    appendStimulusProbe(instances, stimulus);
                }

                updateFlyAgent(fly, stimulus, dt, state.agentEnabled);
                appendFlyInstances(instances, fly);

                if (instances.size() > static_cast<size_t>(kMaxInstances)) {
                    throw std::runtime_error("Scene exceeded kMaxInstances");
                }

                glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
                glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(instances.size() * sizeof(Instance)), instances.data());
            }

            const float cp = std::cos(state.pitch);
            const glm::vec3 cameraPosition(
                state.distance * cp * std::sin(state.yaw),
                state.distance * std::sin(state.pitch),
                state.distance * cp * std::cos(state.yaw));

            glm::vec3 lookTarget(0.0f);
            if (state.sceneMode == 1) lookTarget = office.focusTarget;
            else if (state.sceneMode == 2) lookTarget = kitchen.focusTarget;

            const glm::mat4 view = glm::lookAt(cameraPosition, lookTarget, glm::vec3(0.0f, 1.0f, 0.0f));
            const float aspect = static_cast<float>(state.framebufferW) / static_cast<float>(std::max(state.framebufferH, 1));
            const glm::mat4 projection = glm::perspective(45.0f * kPi / 180.0f, aspect, 0.05f, 100.0f);

            if (state.enhancedLighting && state.sceneMode != 3) {
                glClearColor(0.030f, 0.035f, 0.040f, 1.0f);
            } else {
                glClearColor(0.018f, 0.024f, 0.032f, 1.0f);
            }
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUseProgram(program);
            glUniformMatrix4fv(locView, 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(locProjection, 1, GL_FALSE, glm::value_ptr(projection));
            glUniform3fv(locCameraPosition, 1, glm::value_ptr(cameraPosition));
            glUniform1i(locSceneMode, state.sceneMode);
            glUniform1i(locEnhancedLighting, state.enhancedLighting ? 1 : 0);

            glBindVertexArray(vao);
            glDrawArraysInstanced(GL_TRIANGLES, 0, 36, static_cast<GLsizei>(instances.size()));
            glBindVertexArray(0);
            glfwSwapBuffers(window);

            titleAccumulator += dt;
            ++titleFrames;
            if (titleAccumulator >= 0.5) {
                const double fps = static_cast<double>(titleFrames) / titleAccumulator;
                const int stimulusPercent = static_cast<int>(std::round(stimulus.activity * 100.0f));
                const int neuralPercent = static_cast<int>(std::round(fly.lobula * 100.0f));

                std::string source = sceneModeName(state.sceneMode);
                if (state.sceneMode == 3) {
                    const bool realDepth = state.depthEnabled && depthEstimator.available();
                    source += realDepth ? " DEPTH" : " FALLBACK";
                    if (realDepth) source += " " + std::to_string(static_cast<int>(depthEstimator.lastInferenceMs())) + "ms";
                }

                const std::string title =
                    "FlyVoxelisedReality | M2.6 " + source + " | " +
                    (state.enhancedLighting ? "LIGHTING" : "LEGACY") + " | " +
                    std::to_string(static_cast<int>(fps)) + " FPS | " +
                    std::to_string(instances.size()) + " voxels | stimulus " +
                    std::to_string(stimulusPercent) + "% | pathway " + std::to_string(neuralPercent) + "%" +
                    (state.agentEnabled ? "" : " | AGENT PAUSED") +
                    (state.paused ? " | PAUSED" : "");
                glfwSetWindowTitle(window, title.c_str());
                titleAccumulator = 0.0;
                titleFrames = 0;
            }
        }

        if (capture.isOpened()) capture.release();
        glDeleteBuffers(1, &instanceVbo);
        glDeleteBuffers(1, &cubeVbo);
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(program);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        glfwTerminate();
        return 1;
    }
}
