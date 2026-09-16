#include "depth_estimator.hpp"

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
constexpr int kExtraInstanceCapacity = 16;
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
    bool paused = false;
    bool dragging = false;
    bool depthEnabled = true;
    bool agentEnabled = true;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    float yaw = 0.0f;
    float pitch = 0.05f;
    float distance = 7.2f;
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

out vec3 vNormal;
out vec3 vColor;
out float vMotion;

void main() {
    vec3 world = iPositionScale.xyz + aPosition * iPositionScale.w;
    vNormal = aNormal;
    vColor = iColorMotion.rgb;
    vMotion = iColorMotion.a;
    gl_Position = uProjection * uView * vec4(world, 1.0);
}
)GLSL";

    static constexpr const char* kFragmentShader = R"GLSL(
#version 330 core
in vec3 vNormal;
in vec3 vColor;
in float vMotion;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(vec3(-0.45, 0.75, 0.55));
    float diffuse = max(dot(N, L), 0.0);

    float activity = clamp(vMotion, 0.0, 1.0);
    vec3 activityColour = vec3(0.16, 0.95, 0.72);
    vec3 base = mix(vColor, activityColour, activity * 0.55);
    float rim = pow(1.0 - abs(N.z), 3.0) * 0.08;
    vec3 lit = base * (0.30 + 0.70 * diffuse) + activityColour * rim * activity;
    FragColor = vec4(lit, 1.0);
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
    if (key == GLFW_KEY_R) {
        state->yaw = 0.0f;
        state->pitch = 0.05f;
        state->distance = 7.2f;
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
    state->distance = std::clamp(state->distance - static_cast<float>(yoffset) * 0.45f, 2.4f, 16.0f);
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
    instances.reserve(static_cast<size_t>(kGridW * kGridH + kExtraInstanceCapacity));

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

    // This movement is intentionally an engineering proxy for M2, not a claim
    // about a specific Drosophila behavioural policy. M3 will replace the proxy
    // pathway with connectivity grounded in a real visual-neural map.
    const float speed = 0.10f + 0.85f * fly.lobula;
    fly.position += fly.forward * std::min(speed * dt, distance * 0.18f);

    fly.position.x = std::clamp(fly.position.x, -4.0f, 4.0f);
    fly.position.y = std::clamp(fly.position.y, -3.0f, 3.0f);
    fly.position.z = std::clamp(fly.position.z, -2.5f, 2.8f);
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

        GLFWwindow* window = glfwCreateWindow(1280, 720, "FlyVoxelisedReality | M2 AGENT", nullptr, nullptr);
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
            throw std::runtime_error("GLEW initialisation failed: " + std::string(reinterpret_cast<const char*>(glewGetErrorString(glewResult))));
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

        const std::string depthModelPath = resolveDepthModelPath(argc, argv);
        DepthEstimator depthEstimator(depthModelPath);
        std::cout << "Depth:  " << depthEstimator.status() << '\n';
        if (!depthEstimator.available()) std::cout << "        Run scripts/download_depth_model.ps1, then restart for M1 depth.\n";
        std::cout << "M2:     realtime motion stimulus -> retina -> lamina -> medulla -> lobula -> prototype steering\n";
        std::cout << "Keys:   A agent on/off | D depth/fallback | Space pause | R reset view | drag orbit | wheel zoom\n";

        const GLuint program = createProgram();

        static constexpr float cubeVertices[] = {
            -1,-1,-1,  0, 0,-1,   1, 1,-1,  0, 0,-1,   1,-1,-1,  0, 0,-1,
             1, 1,-1,  0, 0,-1,  -1,-1,-1,  0, 0,-1,  -1, 1,-1,  0, 0,-1,
            -1,-1, 1,  0, 0, 1,   1,-1, 1,  0, 0, 1,   1, 1, 1,  0, 0, 1,
             1, 1, 1,  0, 0, 1,  -1, 1, 1,  0, 0, 1,  -1,-1, 1,  0, 0, 1,
            -1, 1, 1, -1, 0, 0,  -1, 1,-1, -1, 0, 0,  -1,-1,-1, -1, 0, 0,
            -1,-1,-1, -1, 0, 0,  -1,-1, 1, -1, 0, 0,  -1, 1, 1, -1, 0, 0,
             1, 1, 1,  1, 0, 0,   1,-1,-1,  1,0,0,     1,1,-1,   1,0,0,
             1,-1,-1,  1,0,0,     1,1,1,     1,0,0,     1,-1,1,   1,0,0,
            -1,-1,-1,  0,-1,0,    1,-1,-1,   0,-1,0,    1,-1,1,   0,-1,0,
             1,-1,1,   0,-1,0,   -1,-1,1,    0,-1,0,   -1,-1,-1, 0,-1,0,
            -1,1,-1,   0,1,0,     1,1,1,      0,1,0,     1,1,-1,   0,1,0,
             1,1,1,    0,1,0,    -1,1,-1,     0,1,0,    -1,1,1,    0,1,0
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
        glBufferData(GL_ARRAY_BUFFER, (kGridW * kGridH + kExtraInstanceCapacity) * sizeof(Instance), nullptr, GL_STREAM_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), nullptr);
        glVertexAttribDivisor(2, 1);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), reinterpret_cast<void*>(sizeof(glm::vec4)));
        glVertexAttribDivisor(3, 1);
        glBindVertexArray(0);

        glEnable(GL_DEPTH_TEST);
        glClearColor(0.018f, 0.024f, 0.032f, 1.0f);

        cv::VideoCapture capture(0, cv::CAP_ANY);
        if (capture.isOpened()) {
            capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            capture.set(cv::CAP_PROP_FPS, 30);
            std::cout << "Camera: live webcam input\n";
        } else {
            std::cerr << "Camera unavailable; using synthetic fallback input.\n";
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

            if (!state.paused) {
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
                updateFlyAgent(fly, stimulus, dt, state.agentEnabled);

                instances = frameToVoxels(small, flow, depthNearness);
                appendFlyInstances(instances, fly);

                glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
                glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(instances.size() * sizeof(Instance)), instances.data());
                ++capturedFrameIndex;
            }

            const float cp = std::cos(state.pitch);
            const glm::vec3 cameraPosition(
                state.distance * cp * std::sin(state.yaw),
                state.distance * std::sin(state.pitch),
                state.distance * cp * std::cos(state.yaw));

            const glm::mat4 view = glm::lookAt(cameraPosition, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            const float aspect = static_cast<float>(state.framebufferW) / static_cast<float>(std::max(state.framebufferH, 1));
            const glm::mat4 projection = glm::perspective(45.0f * kPi / 180.0f, aspect, 0.05f, 100.0f);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glUseProgram(program);
            glUniformMatrix4fv(glGetUniformLocation(program, "uView"), 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(glGetUniformLocation(program, "uProjection"), 1, GL_FALSE, glm::value_ptr(projection));
            glBindVertexArray(vao);
            glDrawArraysInstanced(GL_TRIANGLES, 0, 36, static_cast<GLsizei>(instances.size()));
            glBindVertexArray(0);
            glfwSwapBuffers(window);

            titleAccumulator += dt;
            ++titleFrames;
            if (titleAccumulator >= 0.5) {
                const double fps = static_cast<double>(titleFrames) / titleAccumulator;
                const bool realDepth = state.depthEnabled && depthEstimator.available();
                std::string depthLabel = realDepth ? "DEPTH" : "FALLBACK";
                if (realDepth) depthLabel += " " + std::to_string(static_cast<int>(depthEstimator.lastInferenceMs())) + "ms";

                const int stimulusPercent = static_cast<int>(std::round(stimulus.activity * 100.0f));
                const int neuralPercent = static_cast<int>(std::round(fly.lobula * 100.0f));
                const std::string title =
                    "FlyVoxelisedReality | M2 AGENT | " + depthLabel + " | " +
                    std::to_string(static_cast<int>(fps)) + " FPS | stimulus " +
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
