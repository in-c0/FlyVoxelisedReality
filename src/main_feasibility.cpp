#include "depth_estimator.hpp"
#include "preset_scenes.hpp"
#include "neural_model.hpp"

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
    bool dashboardEnabled = true;
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
    vec3 groundColour = vec3(0.28, 0.22, 0.17);

    // Hemisphere ambient keeps wall-facing surfaces readable while retaining form.
    float hemi = N.y * 0.5 + 0.5;
    vec3 ambient = mix(groundColour, skyColour, hemi) * 0.66;

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
    float warmBias = max(base.r - base.b, 0.0);
    float greenBias = max(base.g - max(base.r, base.b), 0.0);
    float woodLike = smoothstep(0.08, 0.30, warmBias) * (1.0 - glassLike);
    float leafLike = smoothstep(0.05, 0.28, greenBias);
    float specularStrength = 0.035 + glassLike * 0.58 + darkManufactured * 0.16 +
                             woodLike * 0.035 + leafLike * 0.025;
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
    // Soft floor proximity term darkens contact zones without drawing explicit shadow tiles.
    float horizontalContact = exp(-floorDistance * 3.2) * (0.35 + 0.65 * (1.0 - abs(N.y)));
    float ambientOcclusion = clamp(contactAO * (1.0 - 0.11 * downFacing) *
                                   (1.0 - 0.10 * horizontalContact), 0.64, 1.0);

    vec3 colour = base * (ambient * ambientOcclusion + direct) + sunColour * specular;

    // Preserve live neural activity as a restrained emissive cue.
    colour += activityColour * activity * 0.28;

    // Gentle depth haze separates foreground/midground/background without hiding voxels.
    float cameraDistance = length(uCameraPosition - vWorldPosition);
    float fog = 1.0 - exp(-0.0024 * cameraDistance * cameraDistance);
    vec3 fogColour = uSceneMode == 3 ? vec3(0.035, 0.050, 0.065) : vec3(0.10, 0.115, 0.12);
    colour = mix(colour, fogColour, clamp(fog, 0.0, 0.32));

    // Exposure + filmic tone mapping + display gamma.
    colour *= 1.16;
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
    if (key == GLFW_KEY_TAB) state->dashboardEnabled = !state->dashboardEnabled;
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


VisionFeatures extractVisionFeatures(
    const cv::Mat& flow,
    const cv::Mat& currentGray,
    const cv::Mat& previousGray) {

    VisionFeatures f;
    if (flow.empty()) return f;

    double left = 0.0, right = 0.0, up = 0.0, down = 0.0;
    double radial = 0.0, totalMotion = 0.0;
    double weightedX = 0.0, weightedY = 0.0;

    const float cx = (flow.cols - 1) * 0.5f;
    const float cy = (flow.rows - 1) * 0.5f;

    for (int y = 0; y < flow.rows; ++y) {
        const cv::Point2f* row = flow.ptr<cv::Point2f>(y);
        for (int x = 0; x < flow.cols; ++x) {
            const cv::Point2f v = row[x];
            const float mag = std::sqrt(v.x * v.x + v.y * v.y);
            if (mag < 0.05f) continue;

            left += std::max(-v.x, 0.0f);
            right += std::max(v.x, 0.0f);
            up += std::max(-v.y, 0.0f);
            down += std::max(v.y, 0.0f);
            totalMotion += mag;
            weightedX += x * mag;
            weightedY += y * mag;

            const float rx = x - cx;
            const float ry = y - cy;
            const float r = std::sqrt(rx * rx + ry * ry);
            if (r > 2.0f) radial += std::max((v.x * rx + v.y * ry) / r, 0.0f);
        }
    }

    const double norm = std::max(1.0, static_cast<double>(flow.rows * flow.cols) * 0.32);
    f.motionLeft = std::clamp(static_cast<float>(left / norm), 0.0f, 1.0f);
    f.motionRight = std::clamp(static_cast<float>(right / norm), 0.0f, 1.0f);
    f.motionUp = std::clamp(static_cast<float>(up / norm), 0.0f, 1.0f);
    f.motionDown = std::clamp(static_cast<float>(down / norm), 0.0f, 1.0f);
    f.looming = std::clamp(static_cast<float>(radial / (norm * 0.70)), 0.0f, 1.0f);
    f.confidence = std::clamp(static_cast<float>(totalMotion / (norm * 1.2)), 0.0f, 1.0f);

    if (totalMotion > 0.5) {
        const float ox = static_cast<float>(weightedX / totalMotion);
        const float oy = static_cast<float>(weightedY / totalMotion);
        f.objectX = std::clamp((ox - cx) / std::max(cx, 1.0f), -1.0f, 1.0f);
        f.objectY = std::clamp((cy - oy) / std::max(cy, 1.0f), -1.0f, 1.0f);
    }

    if (!previousGray.empty() && previousGray.size() == currentGray.size()) {
        double onSum = 0.0, offSum = 0.0;
        for (int y = 0; y < currentGray.rows; ++y) {
            const unsigned char* now = currentGray.ptr<unsigned char>(y);
            const unsigned char* prev = previousGray.ptr<unsigned char>(y);
            for (int x = 0; x < currentGray.cols; ++x) {
                const int d = static_cast<int>(now[x]) - static_cast<int>(prev[x]);
                if (d > 0) onSum += d;
                else offSum -= d;
            }
        }
        const double lumNorm = std::max(1.0, static_cast<double>(currentGray.total()) * 22.0);
        f.on = std::clamp(static_cast<float>(onSum / lumNorm), 0.0f, 1.0f);
        f.off = std::clamp(static_cast<float>(offSum / lumNorm), 0.0f, 1.0f);
    } else {
        f.on = f.off = f.confidence * 0.35f;
    }

    return f;
}

VisionFeatures makePresetVisionFeatures(double elapsed, int sceneMode) {
    const float t = static_cast<float>(elapsed);
    const float phase = sceneMode == 1 ? 0.0f : 1.1f;
    const float vx = std::cos(t * 0.75f + phase);
    const float vy = std::cos(t * 1.15f + phase);
    const float pulse = std::max(std::sin(t * 0.55f + phase), 0.0f);
    const float looming = std::pow(pulse, 7.0f);

    VisionFeatures f;
    f.motionLeft = vx < 0.0f ? std::abs(vx) * 0.70f : 0.0f;
    f.motionRight = vx >= 0.0f ? std::abs(vx) * 0.70f : 0.0f;
    f.motionUp = vy < 0.0f ? std::abs(vy) * 0.28f : 0.0f;
    f.motionDown = vy >= 0.0f ? std::abs(vy) * 0.28f : 0.0f;
    f.looming = looming;
    f.on = 0.28f + 0.20f * (0.5f + 0.5f * std::sin(t * 1.4f));
    f.off = 0.24f + 0.18f * (0.5f + 0.5f * std::cos(t * 1.1f));
    f.objectX = std::sin(t * 0.75f + phase) * 0.75f;
    f.objectY = std::sin(t * 1.15f + phase) * 0.25f;
    f.confidence = 0.82f;
    return f;
}

void updateFlyFromMotor(FlyAgent& fly, const MotorCommand& motor, float dt, bool enabled) {
    if (!enabled) return;
    glm::vec3 forward = glm::length(fly.forward) > 0.001f ? glm::normalize(fly.forward)
                                                          : glm::vec3(0.0f, 0.0f, -1.0f);
    const float yaw = motor.yaw * dt * (1.4f + 2.6f * motor.escape);
    const float c = std::cos(yaw);
    const float sn = std::sin(yaw);
    glm::vec3 rotated(c * forward.x - sn * forward.z,
                      forward.y,
                      sn * forward.x + c * forward.z);
    rotated.y = std::clamp(rotated.y + motor.pitch * dt * 0.9f, -0.65f, 0.65f);
    fly.forward = glm::normalize(rotated);

    const float speed = 0.20f + motor.forward * (0.55f + 1.35f * motor.escape);
    fly.position += fly.forward * speed * dt;
    fly.position.x = std::clamp(fly.position.x, -3.8f, 3.8f);
    fly.position.y = std::clamp(fly.position.y, -1.35f, 2.4f);
    fly.position.z = std::clamp(fly.position.z, -2.55f, 2.8f);
}

GLuint createBlitProgram() {
    static constexpr const char* vsSource = R"GLSL(
#version 330 core
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
)GLSL";
    static constexpr const char* fsSource = R"GLSL(
#version 330 core
in vec2 vUV;
uniform sampler2D uTexture;
out vec4 FragColor;
void main() {
    FragColor = texture(uTexture, vUV);
}
)GLSL";
    const GLuint vs = compileShader(GL_VERTEX_SHADER, vsSource);
    const GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSource);
    const GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) throw std::runtime_error("Dashboard shader link failed");
    return p;
}

cv::Scalar activationColour(float a) {
    a = std::clamp(a, 0.0f, 1.0f);
    return cv::Scalar(55.0 + 20.0 * (1.0 - a),
                      80.0 + 115.0 * a,
                      95.0 + 160.0 * a);
}

cv::Mat depthByteFromNearness(const cv::Mat& nearness) {
    cv::Mat out;
    if (nearness.empty()) return cv::Mat(120, 160, CV_8UC1, cv::Scalar(0));
    cv::Mat clamped;
    cv::max(nearness, 0.0, clamped);
    cv::min(clamped, 1.0, clamped);
    clamped.convertTo(out, CV_8UC1, 255.0);
    return out;
}

cv::Mat depthByteFromGL(const cv::Mat& depth) {
    cv::Mat inv = 1.0f - depth;
    cv::Mat mask = depth < 0.9995f;
    double minV = 0.0, maxV = 1.0;
    cv::minMaxLoc(inv, &minV, &maxV, nullptr, nullptr, mask);
    cv::Mat norm = cv::Mat::zeros(depth.size(), CV_32F);
    if (maxV > minV + 1e-6) norm = (inv - minV) / (maxV - minV);
    norm.setTo(0.0f, ~mask);
    cv::Mat out;
    norm.convertTo(out, CV_8UC1, 255.0);
    return out;
}

cv::Mat colouriseDepth(const cv::Mat& depthByte) {
    cv::Mat c;
    cv::applyColorMap(depthByte, c, cv::COLORMAP_TURBO);
    return c;
}

cv::Mat makeRegionSegmentation(const cv::Mat& depthByte) {
    cv::Mat seg(depthByte.size(), CV_8UC3, cv::Scalar(18, 22, 28));
    for (int y = 0; y < depthByte.rows; ++y) {
        const unsigned char* d = depthByte.ptr<unsigned char>(y);
        cv::Vec3b* o = seg.ptr<cv::Vec3b>(y);
        for (int x = 0; x < depthByte.cols; ++x) {
            if (d[x] < 20) o[x] = {18, 22, 28};
            else if (d[x] < 90) o[x] = {110, 72, 38};
            else if (d[x] < 175) o[x] = {65, 150, 95};
            else o[x] = {48, 118, 220};
        }
    }
    return seg;
}

cv::Mat makeMotionViz(const cv::Mat& flow, const VisionFeatures& f) {
    cv::Mat vis(140, 280, CV_8UC3, cv::Scalar(12, 16, 22));
    if (!flow.empty()) {
        cv::Mat mag(flow.size(), CV_32F);
        for (int y = 0; y < flow.rows; ++y) {
            const cv::Point2f* row = flow.ptr<cv::Point2f>(y);
            float* m = mag.ptr<float>(y);
            for (int x = 0; x < flow.cols; ++x) {
                m[x] = std::sqrt(row[x].x * row[x].x + row[x].y * row[x].y);
            }
        }
        double mx = 0.0;
        cv::minMaxLoc(mag, nullptr, &mx);
        cv::Mat m8;
        mag.convertTo(m8, CV_8UC1, mx > 1e-5 ? 255.0 / mx : 0.0);
        cv::Mat colour;
        cv::applyColorMap(m8, colour, cv::COLORMAP_VIRIDIS);
        cv::resize(colour, vis, vis.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    const cv::Point center(vis.cols / 2, vis.rows / 2);
    cv::Point end(center.x + static_cast<int>((f.motionRight - f.motionLeft) * 90.0f),
                  center.y + static_cast<int>((f.motionDown - f.motionUp) * 50.0f));
    cv::arrowedLine(vis, center, end, cv::Scalar(255, 255, 255), 2, cv::LINE_AA, 0, 0.22);
    cv::putText(vis, "loom " + std::to_string(static_cast<int>(f.looming * 100.0f)) + "%",
                {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(230,230,230), 1, cv::LINE_AA);
    return vis;
}

cv::Mat makeNeuralMapViz(const NeuralState& n) {
    cv::Mat img(300, 440, CV_8UC3, cv::Scalar(9, 13, 19));
    struct Node { const char* name; cv::Point p; float a; };
    const float t4 = *std::max_element(n.t4.begin(), n.t4.end());
    const float t5 = *std::max_element(n.t5.begin(), n.t5.end());
    std::vector<Node> nodes = {
        {"R1-6",{35,145},n.r1r6}, {"L1",{100,105},n.l1}, {"L2",{100,190},n.l2},
        {"Mi1",{165,70},n.mi1}, {"Tm3",{165,120},n.tm3},
        {"Tm1",{165,175},n.tm1}, {"Tm2",{165,225},n.tm2},
        {"T4",{240,95},t4}, {"T5",{240,200},t5},
        {"HS/VS",{315,120},std::max(n.hs,n.vs)},
        {"LC4",{315,195},n.lc4}, {"LPLC2",{315,245},n.lplc2},
        {"GF",{395,220},n.giantFiber}
    };
    auto line = [&](int a, int b) {
        cv::line(img, nodes[a].p, nodes[b].p, cv::Scalar(58,72,88), 1, cv::LINE_AA);
    };
    line(0,1); line(0,2); line(1,3); line(1,4); line(2,5); line(2,6);
    line(3,7); line(4,7); line(5,8); line(6,8); line(7,9); line(8,9);
    line(8,10); line(10,12); line(11,12);
    for (const auto& node : nodes) {
        cv::circle(img, node.p, 10, activationColour(node.a), cv::FILLED, cv::LINE_AA);
        cv::putText(img, node.name, node.p + cv::Point(-14,-14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.36, cv::Scalar(220,225,230), 1, cv::LINE_AA);
    }
    cv::putText(img, "literature topology / proxy dynamics", {10,286},
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(130,145,160), 1, cv::LINE_AA);
    return img;
}

cv::Mat applyLearnedNeuralRender(const cv::Mat& bgr, const NeuralState& n) {
    if (bgr.empty()) return cv::Mat(160, 280, CV_8UC3, cv::Scalar(0,0,0));
    cv::Mat small;
    cv::resize(bgr, small, cv::Size(320, 180), 0.0, 0.0, cv::INTER_AREA);

    static const float W1[7][10] = {
        {0.850268f,0.366639f,0.330808f,-0.836603f,-0.357363f,0.526599f,-0.187690f,0.338803f,1.239338f,-0.017944f},
        {-0.722003f,-1.286454f,-0.093354f,-0.327576f,-0.051875f,0.305599f,1.107628f,-0.202084f,0.239478f,-0.357051f},
        {-0.142119f,0.520022f,-0.969060f,0.785718f,1.199622f,0.321120f,0.140141f,0.701481f,0.311006f,-0.337428f},
        {-0.340996f,0.194262f,0.190417f,0.074951f,-0.033756f,0.187376f,-0.105882f,-0.242287f,-0.069385f,-0.094693f},
        {0.212667f,-0.011147f,0.128654f,0.039386f,-0.353517f,-0.161602f,-0.033973f,-0.022775f,0.215369f,-0.087658f},
        {-0.128413f,0.083744f,0.196969f,-0.215969f,0.492600f,-0.209055f,-0.353798f,-0.265379f,-0.157187f,0.131319f},
        {0.077369f,-0.115708f,0.239044f,0.204398f,0.106113f,0.221429f,-0.302197f,0.442782f,-0.313633f,-0.067725f}
    };
    static const float B1[10] = {0.114220f,0.072466f,-0.005995f,0.058808f,-0.409004f,-0.310571f,-0.092330f,-0.209554f,-0.575465f,0.125902f};
    static const float W2[10][3] = {
        {0.726964f,-0.882471f,-0.100451f},{0.501341f,-1.336719f,0.579698f},
        {0.473795f,0.051861f,-0.741476f},{-0.956418f,-0.331518f,0.813448f},
        {-0.032610f,0.180955f,1.363186f},{0.414066f,0.435338f,0.323225f},
        {-0.190919f,0.873074f,0.160229f},{0.473467f,-0.089789f,0.778591f},
        {1.478028f,0.508548f,0.591808f},{-0.195741f,-0.306296f,-0.360099f}
    };
    static const float B2[3] = {-0.082057f,-0.209945f,-0.332963f};

    const float motion = std::max({n.t4[0],n.t4[1],n.t4[2],n.t4[3],n.t5[0],n.t5[1],n.t5[2],n.t5[3]});
    const float extra[4] = {motion, n.giantFiber, n.l1, n.l2};

    cv::Mat out(small.size(), CV_8UC3);
    for (int y = 0; y < small.rows; ++y) {
        const cv::Vec3b* src = small.ptr<cv::Vec3b>(y);
        cv::Vec3b* dst = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < small.cols; ++x) {
            float input[7] = {
                src[x][2] / 255.0f, src[x][1] / 255.0f, src[x][0] / 255.0f,
                extra[0], extra[1], extra[2], extra[3]
            };
            float h[10];
            for (int j = 0; j < 10; ++j) {
                float z = B1[j];
                for (int i = 0; i < 7; ++i) z += input[i] * W1[i][j];
                h[j] = std::tanh(z);
            }
            float rgb[3];
            for (int k = 0; k < 3; ++k) {
                float z = B2[k];
                for (int j = 0; j < 10; ++j) z += h[j] * W2[j][k];
                rgb[k] = 1.0f / (1.0f + std::exp(-z));
            }
            dst[x] = cv::Vec3b(
                static_cast<unsigned char>(std::clamp(rgb[2],0.0f,1.0f) * 255.0f),
                static_cast<unsigned char>(std::clamp(rgb[1],0.0f,1.0f) * 255.0f),
                static_cast<unsigned char>(std::clamp(rgb[0],0.0f,1.0f) * 255.0f));
        }
    }
    return out;
}

cv::Mat makeBehaviourViz(
    const std::vector<glm::vec3>& path,
    const FlyAgent& fly,
    const MotorCommand& motor) {

    cv::Mat img(300, 440, CV_8UC3, cv::Scalar(10,14,18));
    auto mapPoint = [](const glm::vec3& p) {
        return cv::Point(static_cast<int>(220 + p.x * 45.0f),
                         static_cast<int>(155 + p.z * 42.0f));
    };
    for (size_t i = 1; i < path.size(); ++i) {
        cv::line(img, mapPoint(path[i-1]), mapPoint(path[i]), cv::Scalar(85,135,185), 2, cv::LINE_AA);
    }
    const cv::Point p = mapPoint(fly.position);
    cv::circle(img, p, 7, cv::Scalar(50,190,240), cv::FILLED, cv::LINE_AA);
    cv::Point dir(p.x + static_cast<int>(fly.forward.x * 42.0f),
                  p.y + static_cast<int>(fly.forward.z * 42.0f));
    cv::arrowedLine(img, p, dir, cv::Scalar(235,235,235), 2, cv::LINE_AA, 0, 0.25);
    cv::putText(img, "yaw " + std::to_string(motor.yaw).substr(0,5), {12,25},
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(220,225,230), 1, cv::LINE_AA);
    cv::putText(img, "escape " + std::to_string(static_cast<int>(motor.escape*100)) + "%", {12,47},
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(220,225,230), 1, cv::LINE_AA);
    return img;
}

void drawBar(cv::Mat& img, int x, int y, int w, const char* label, float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    cv::putText(img, label, {x,y-3}, cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cv::Scalar(190,198,208), 1, cv::LINE_AA);
    cv::rectangle(img, {x,y+3,w,10}, cv::Scalar(45,52,62), cv::FILLED);
    cv::rectangle(img, {x,y+3,static_cast<int>(w*value),10}, activationColour(value), cv::FILLED);
}

cv::Mat makeCausalTrace(const VisionFeatures& f, const NeuralState& n, const MotorCommand& m) {
    cv::Mat img(95, 1280, CV_8UC3, cv::Scalar(8,12,17));
    cv::putText(img, "stimulus", {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(230,235,240), 1, cv::LINE_AA);
    drawBar(img, 10, 40, 120, "motion", std::max({f.motionLeft,f.motionRight,f.motionUp,f.motionDown}));
    drawBar(img, 145, 40, 120, "loom", f.looming);
    drawBar(img, 280, 40, 120, "L1", n.l1);
    drawBar(img, 415, 40, 120, "L2", n.l2);
    drawBar(img, 550, 40, 120, "T4", *std::max_element(n.t4.begin(),n.t4.end()));
    drawBar(img, 685, 40, 120, "T5", *std::max_element(n.t5.begin(),n.t5.end()));
    drawBar(img, 820, 40, 120, "LC4", n.lc4);
    drawBar(img, 955, 40, 120, "GF", n.giantFiber);
    drawBar(img, 1090, 40, 120, "motor", std::max(std::abs(m.yaw),m.escape));
    cv::putText(img, "CV -> ON/OFF + motion/looming -> mapped visual types -> proxy motor readout",
                {280,20}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(150,165,180), 1, cv::LINE_AA);
    return img;
}

void placePanel(cv::Mat& dashboard, const cv::Rect& rect, const cv::Mat& image, const std::string& title) {
    cv::rectangle(dashboard, rect, cv::Scalar(47,58,68), 1);
    cv::rectangle(dashboard, cv::Rect(rect.x, rect.y, rect.width, 25), cv::Scalar(18,25,32), cv::FILLED);
    cv::putText(dashboard, title, {rect.x+8,rect.y+18}, cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(225,230,235), 1, cv::LINE_AA);
    if (image.empty() || rect.height <= 27) return;
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(rect.width-2, rect.height-27), 0.0, 0.0, cv::INTER_AREA);
    resized.copyTo(dashboard(cv::Rect(rect.x+1,rect.y+26,rect.width-2,rect.height-27)));
}

cv::Mat composeDashboard(
    const cv::Mat& source,
    const cv::Mat& depthByte,
    const cv::Mat& segmentation,
    const cv::Mat& motion,
    const cv::Mat& scene,
    const cv::Mat& neuralMap,
    const cv::Mat& neuralRender,
    const cv::Mat& behaviour,
    const cv::Mat& trace) {

    cv::Mat dash(720, 1280, CV_8UC3, cv::Scalar(6,9,13));
    const int topH = 160;
    const int bottomH = 95;
    const int midH = 720 - topH - bottomH;
    for (int i = 0; i < 4; ++i) {
        cv::Rect r(i*320,0,320,topH);
        if (i==0) placePanel(dash,r,source,"1 SOURCE RGB");
        if (i==1) placePanel(dash,r,colouriseDepth(depthByte),"2 DEPTH");
        if (i==2) placePanel(dash,r,segmentation,"3 REGION SEGMENT");
        if (i==3) placePanel(dash,r,motion,"4 MOTION / LOOMING");
    }
    placePanel(dash, cv::Rect(0,topH,850,midH), scene, "5 MAIN VOXEL WORLD");
    const int sideX = 850;
    const int sideW = 430;
    const int h0 = midH / 3;
    placePanel(dash, cv::Rect(sideX,topH,sideW,h0), neuralMap, "6 FLY NEURAL MAP");
    placePanel(dash, cv::Rect(sideX,topH+h0,sideW,h0), neuralRender, "7 LEARNED NEURAL RENDER");
    placePanel(dash, cv::Rect(sideX,topH+2*h0,sideW,midH-2*h0), behaviour, "8 BEHAVIOUR");
    placePanel(dash, cv::Rect(0,topH+midH,1280,bottomH), trace, "9 CAUSAL TRACE");
    return dash;
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

        GLFWwindow* window = glfwCreateWindow(1280, 720, "FlyVoxelisedReality | M3-M7 FEASIBILITY", nullptr, nullptr);
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
        std::cout << "M3-M7:  mapped visual circuit + CV features + behaviour + dashboard + learned render\n";
        std::cout << "Keys:   1 office | 2 kitchen | 3 live CV | Tab dashboard | L lighting | A agent | D depth | Space pause | R reset\n";

        const GLuint program = createProgram();
        const GLint locView = glGetUniformLocation(program, "uView");
        const GLint locProjection = glGetUniformLocation(program, "uProjection");
        const GLint locCameraPosition = glGetUniformLocation(program, "uCameraPosition");
        const GLint locSceneMode = glGetUniformLocation(program, "uSceneMode");
        const GLint locEnhancedLighting = glGetUniformLocation(program, "uEnhancedLighting");
        const GLuint blitProgram = createBlitProgram();
        const GLint locBlitTexture = glGetUniformLocation(blitProgram, "uTexture");

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

        // Offscreen world render. The dashboard reuses this one render instead of
        // drawing nine independent 3D worlds.
        constexpr int kSceneRenderW = 640;
        constexpr int kSceneRenderH = 360;
        GLuint sceneFbo = 0;
        GLuint sceneColourTex = 0;
        GLuint sceneDepthRbo = 0;
        glGenFramebuffers(1, &sceneFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo);
        glGenTextures(1, &sceneColourTex);
        glBindTexture(GL_TEXTURE_2D, sceneColourTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, kSceneRenderW, kSceneRenderH, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneColourTex, 0);
        glGenRenderbuffers(1, &sceneDepthRbo);
        glBindRenderbuffer(GL_RENDERBUFFER, sceneDepthRbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, kSceneRenderW, kSceneRenderH);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, sceneDepthRbo);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Scene framebuffer is incomplete");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Fullscreen dashboard blit.
        static constexpr float quadVertices[] = {
            -1.0f,-1.0f, 0.0f,1.0f,   1.0f,-1.0f, 1.0f,1.0f,   1.0f,1.0f, 1.0f,0.0f,
            -1.0f,-1.0f, 0.0f,1.0f,   1.0f,1.0f, 1.0f,0.0f,  -1.0f,1.0f, 0.0f,0.0f
        };
        GLuint quadVao = 0, quadVbo = 0, dashboardTexture = 0;
        glGenVertexArrays(1, &quadVao);
        glGenBuffers(1, &quadVbo);
        glBindVertexArray(quadVao);
        glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), reinterpret_cast<void*>(2*sizeof(float)));
        glBindVertexArray(0);

        glGenTextures(1, &dashboardTexture);
        glBindTexture(GL_TEXTURE_2D, dashboardTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 1280, 720, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        std::vector<unsigned char> sceneRgb(static_cast<size_t>(kSceneRenderW*kSceneRenderH*3));
        std::vector<float> sceneDepth(static_cast<size_t>(kSceneRenderW*kSceneRenderH));

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
        VisionFeatures vision;
        FlyVisualCircuit circuit;
        MotorCommand motor;
        std::vector<glm::vec3> trajectory;
        resetFlyForMode(fly, state.sceneMode, office, kitchen);
        trajectory.push_back(fly.position);

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
                vision = VisionFeatures{};
                motor = MotorCommand{};
                circuit.reset();
                trajectory.clear();
                trajectory.push_back(fly.position);
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

                    if (state.depthEnabled && depthEstimator.available()) {
                        if (depthNearness.empty() || (capturedFrameIndex % kDepthInferenceInterval) == 0) {
                            cv::Mat inferred = depthEstimator.inferNearness(frame, cv::Size(kGridW, kGridH));
                            if (!inferred.empty()) depthNearness = inferred;
                        }
                    } else {
                        depthNearness = luminanceFallbackNearness(small);
                    }
                    if (depthNearness.empty()) depthNearness = luminanceFallbackNearness(small);

                    vision = extractVisionFeatures(flow, gray, previousGray);
                    gray.copyTo(previousGray);
                    stimulus = extractVisualStimulus(flow, depthNearness);
                    instances = frameToVoxels(small, flow, depthNearness);
                    ++capturedFrameIndex;
                } else {
                    const PresetScene& scene = state.sceneMode == 1 ? office : kitchen;
                    stimulus = makePresetStimulus(scene, elapsed, state.sceneMode);
                    vision = makePresetVisionFeatures(elapsed, state.sceneMode);
                    instances = presetToInstances(scene);
                    appendStimulusProbe(instances, stimulus);
                }

                motor = circuit.update(vision, dt);
                updateFlyFromMotor(fly, motor, dt, state.agentEnabled);
                const NeuralState& neural = circuit.state();
                fly.retina = neural.r1r6;
                fly.lamina = std::max(neural.l1, neural.l2);
                fly.medulla = std::max({neural.mi1, neural.tm3, neural.tm1, neural.tm2});
                fly.lobula = std::max({
                    *std::max_element(neural.t4.begin(), neural.t4.end()),
                    *std::max_element(neural.t5.begin(), neural.t5.end()),
                    neural.lc4, neural.lplc2
                });
                appendFlyInstances(instances, fly);
                if (trajectory.empty() || glm::length(trajectory.back() - fly.position) > 0.06f) {
                    trajectory.push_back(fly.position);
                    if (trajectory.size() > 120) trajectory.erase(trajectory.begin());
                }

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
            const float aspect = static_cast<float>(kSceneRenderW) / static_cast<float>(kSceneRenderH);
            const glm::mat4 projection = glm::perspective(45.0f * kPi / 180.0f, aspect, 0.05f, 100.0f);

            // 5 MAIN WORLD: render once offscreen.
            glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo);
            glViewport(0, 0, kSceneRenderW, kSceneRenderH);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_CULL_FACE);
            if (state.enhancedLighting && state.sceneMode != 3) glClearColor(0.030f, 0.035f, 0.040f, 1.0f);
            else glClearColor(0.018f, 0.024f, 0.032f, 1.0f);
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

            glReadPixels(0, 0, kSceneRenderW, kSceneRenderH, GL_RGB, GL_UNSIGNED_BYTE, sceneRgb.data());
            glReadPixels(0, 0, kSceneRenderW, kSceneRenderH, GL_DEPTH_COMPONENT, GL_FLOAT, sceneDepth.data());
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            cv::Mat sceneRgbMat(kSceneRenderH, kSceneRenderW, CV_8UC3, sceneRgb.data());
            cv::Mat sceneRgbFlipped;
            cv::flip(sceneRgbMat, sceneRgbFlipped, 0);
            cv::Mat sceneBgr;
            cv::cvtColor(sceneRgbFlipped, sceneBgr, cv::COLOR_RGB2BGR);

            cv::Mat depthFloat(kSceneRenderH, kSceneRenderW, CV_32F, sceneDepth.data());
            cv::Mat depthFloatFlipped;
            cv::flip(depthFloat, depthFloatFlipped, 0);
            cv::Mat depthByte = (state.sceneMode == 3 && !depthNearness.empty())
                ? depthByteFromNearness(depthNearness)
                : depthByteFromGL(depthFloatFlipped);

            const cv::Mat sourcePanel = (state.sceneMode == 3 && !frame.empty()) ? frame : sceneBgr;
            const cv::Mat segmentation = makeRegionSegmentation(depthByte);
            const cv::Mat motionPanel = makeMotionViz(flow, vision);
            const NeuralState& neural = circuit.state();
            const cv::Mat neuralMap = makeNeuralMapViz(neural);
            const cv::Mat neuralRender = applyLearnedNeuralRender(sceneBgr, neural);
            const cv::Mat behaviour = makeBehaviourViz(trajectory, fly, motor);
            const cv::Mat trace = makeCausalTrace(vision, neural, motor);
            cv::Mat dashboard = composeDashboard(
                sourcePanel, depthByte, segmentation, motionPanel, sceneBgr,
                neuralMap, neuralRender, behaviour, trace);

            cv::Mat display;
            if (state.dashboardEnabled) display = dashboard;
            else cv::resize(sceneBgr, display, cv::Size(1280,720), 0.0, 0.0, cv::INTER_LINEAR);

            glViewport(0, 0, state.framebufferW, state.framebufferH);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
            glClearColor(0.005f,0.008f,0.012f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(blitProgram);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, dashboardTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1280, 720, GL_BGR, GL_UNSIGNED_BYTE, display.data);
            glUniform1i(locBlitTexture, 0);
            glBindVertexArray(quadVao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_CULL_FACE);
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
                    "FlyVoxelisedReality | M3-M7 " + source + " | " +
                    (state.dashboardEnabled ? "DASHBOARD" : "WORLD") + " | " +
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
        glDeleteTextures(1, &dashboardTexture);
        glDeleteTextures(1, &sceneColourTex);
        glDeleteRenderbuffers(1, &sceneDepthRbo);
        glDeleteFramebuffers(1, &sceneFbo);
        glDeleteBuffers(1, &quadVbo);
        glDeleteVertexArrays(1, &quadVao);
        glDeleteProgram(blitProgram);
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
