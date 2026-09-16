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
constexpr float kPi = 3.14159265358979323846f;
constexpr float kVirtualCameraFovDegrees = 68.0f;
constexpr float kResearcherFovDegrees = 45.0f;
constexpr float kFlyPreviewFovDegrees = 100.0f;
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
    bool flyVisionEnabled = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    float yaw = 0.0f;
    float pitch = 0.05f;
    float distance = 7.2f;
};

struct RenderTarget {
    GLuint framebuffer = 0;
    GLuint colourTexture = 0;
    GLuint depthStencil = 0;
    int width = 0;
    int height = 0;
};

float quantise(float value, float step) {
    return std::round(value / step) * step;
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

GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader) {
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

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

GLuint createSceneProgram() {
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
    const GLuint program = linkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

GLuint createPerceptionProgram() {
    static constexpr const char* kVertexShader = R"GLSL(
#version 330 core
out vec2 vUv;

const vec2 kPositions[6] = vec2[](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main() {
    vec2 p = kPositions[gl_VertexID];
    vUv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
)GLSL";

    static constexpr const char* kFragmentShader = R"GLSL(
#version 330 core
in vec2 vUv;
out vec4 FragColor;

uniform sampler2D uScene;
uniform vec2 uResolution;
uniform int uFlyVision;

vec3 flyInspiredColour(vec3 rgb) {
    // A webcam has no UV channel, so this is deliberately only an RGB proxy:
    // attenuate long-wavelength red and bias visible contrast toward green/blue.
    float blueGreen = 0.58 * rgb.b + 0.42 * rgb.g;
    float green = 0.82 * rgb.g + 0.18 * rgb.b;
    float dimRed = 0.22 * rgb.r + 0.08 * rgb.g;
    return vec3(dimRed, green, blueGreen);
}

void main() {
    if (uFlyVision == 0) {
        FragColor = texture(uScene, vUv);
        return;
    }

    vec2 resolution = max(uResolution, vec2(1.0));
    vec2 pixel = vUv * resolution;

    // Rough ommatidial packing: staggered lens rows, sized from screen height so
    // aspect ratio changes do not stretch the cells. This is a perceptual preview,
    // not a claim of anatomically exact Drosophila optics.
    float cellWidth = max(resolution.y / 23.0, 10.0);
    float rowHeight = cellWidth * 0.8660254;
    float row = floor(pixel.y / rowHeight);
    float rowOffset = mod(row, 2.0) * cellWidth * 0.5;
    float col = floor((pixel.x - rowOffset) / cellWidth);

    vec2 centrePx = vec2(
        (col + 0.5) * cellWidth + rowOffset,
        (row + 0.5) * rowHeight
    );

    vec2 localPx = pixel - centrePx;
    vec2 local = vec2(
        localPx.x / (cellWidth * 0.52),
        localPx.y / (rowHeight * 0.60)
    );

    float radius = length(local);
    vec2 centreUv = centrePx / resolution;

    // Sample a small portion of the image inside each lens rather than reducing
    // every lens to one flat colour. This keeps motion readable while still making
    // the angular sampling obvious.
    vec2 lensOffset = vec2(
        local.x * cellWidth,
        local.y * rowHeight
    ) / resolution * 0.16;
    vec2 sampleUv = clamp(centreUv + lensOffset, vec2(0.0), vec2(1.0));

    vec3 colour = texture(uScene, sampleUv).rgb;
    colour = flyInspiredColour(colour);

    // Lens curvature cue plus dark inter-ommatidial seams.
    float lensShade = 1.08 - 0.20 * clamp(radius * radius, 0.0, 1.0);
    colour *= lensShade;
    float seam = smoothstep(0.82, 1.00, radius);
    colour = mix(colour, vec3(0.006, 0.010, 0.012), seam);

    FragColor = vec4(colour, 1.0);
}
)GLSL";

    const GLuint vs = compileShader(GL_VERTEX_SHADER, kVertexShader);
    const GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    const GLuint program = linkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

void destroyRenderTarget(RenderTarget& target) {
    if (target.depthStencil != 0) glDeleteRenderbuffers(1, &target.depthStencil);
    if (target.colourTexture != 0) glDeleteTextures(1, &target.colourTexture);
    if (target.framebuffer != 0) glDeleteFramebuffers(1, &target.framebuffer);
    target = {};
}

void ensureRenderTarget(RenderTarget& target, int width, int height) {
    width = std::max(width, 1);
    height = std::max(height, 1);
    if (target.framebuffer != 0 && target.width == width && target.height == height) return;

    destroyRenderTarget(target);

    glGenFramebuffers(1, &target.framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);

    glGenTextures(1, &target.colourTexture);
    glBindTexture(GL_TEXTURE_2D, target.colourTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target.colourTexture, 0);

    glGenRenderbuffers(1, &target.depthStencil);
    glBindRenderbuffer(GL_RENDERBUFFER, target.depthStencil);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, target.depthStencil);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        destroyRenderTarget(target);
        throw std::runtime_error("Off-screen perception framebuffer is incomplete");
    }

    target.width = width;
    target.height = height;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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
    if (key == GLFW_KEY_F) state->flyVisionEnabled = !state->flyVisionEnabled;
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
    instances.reserve(static_cast<size_t>(kGridW * kGridH));

    const float halfW = static_cast<float>(kGridW - 1) * 0.5f;
    const float halfH = static_cast<float>(kGridH - 1) * 0.5f;
    const float fovRadians = kVirtualCameraFovDegrees * kPi / 180.0f;
    const float focalPixels = (0.5f * static_cast<float>(kGridW)) / std::tan(fovRadians * 0.5f);
    const float depthMidpoint = (kNearDistance + kFarDistance) * 0.5f;

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

            const float nearValue = depthRow ? std::clamp(depthRow[x], 0.0f, 1.0f) : 0.5f;
            const float distance = kFarDistance - nearValue * (kFarDistance - kNearDistance);

            float worldX = ((static_cast<float>(x) - halfW) / focalPixels) * distance;
            float worldY = ((halfH - static_cast<float>(y)) / focalPixels) * distance;
            float worldZ = -(distance - depthMidpoint);

            worldX = quantise(worldX, kVoxelQuantisation);
            worldY = quantise(worldY, kVoxelQuantisation);
            worldZ = quantise(worldZ, kVoxelQuantisation);

            const float distance01 = (distance - kNearDistance) / (kFarDistance - kNearDistance);
            const float scale = 0.045f + distance01 * 0.030f + motion * 0.015f;

            instances.push_back({
                glm::vec4(worldX, worldY, worldZ, scale),
                glm::vec4(colour, motion)
            });
        }
    }
    return instances;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (!glfwInit()) {
            throw std::runtime_error("GLFW initialisation failed");
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        GLFWwindow* window = glfwCreateWindow(1280, 720, "FlyVoxelisedReality | M2", nullptr, nullptr);
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
            throw std::runtime_error(
                "GLEW initialisation failed: " +
                std::string(reinterpret_cast<const char*>(glewGetErrorString(glewResult))));
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
        if (!depthEstimator.available()) {
            std::cout << "        Run scripts/download_depth_model.ps1, then restart for M1/M2 depth.\n";
        }
        std::cout << "Keys:   F fly/researcher | D depth/fallback | Space pause | R reset | drag orbit | wheel zoom\n";

        const GLuint sceneProgram = createSceneProgram();
        const GLuint perceptionProgram = createPerceptionProgram();

        static constexpr float cubeVertices[] = {
            -1,-1,-1,  0, 0,-1,   1, 1,-1,  0, 0,-1,   1,-1,-1,  0, 0,-1,
             1, 1,-1,  0, 0,-1,  -1,-1,-1,  0, 0,-1,  -1, 1,-1,  0, 0,-1,
            -1,-1, 1,  0, 0, 1,   1,-1, 1,  0, 0, 1,   1, 1, 1,  0, 0, 1,
             1, 1, 1,  0, 0, 1,  -1, 1, 1,  0, 0, 1,  -1,-1, 1,  0, 0, 1,
            -1, 1, 1, -1, 0, 0,  -1, 1,-1, -1, 0, 0,  -1,-1,-1, -1, 0, 0,
            -1,-1,-1, -1, 0, 0,  -1,-1, 1, -1, 0, 0,  -1, 1, 1, -1, 0, 0,
             1, 1, 1,  1, 0, 0,   1,-1,-1,  1, 0, 0,   1, 1,-1,  1, 0, 0,
             1,-1,-1,  1, 0, 0,   1, 1, 1,  1, 0, 0,   1,-1, 1,  1, 0, 0,
            -1,-1,-1,  0,-1, 0,   1,-1,-1,  0,-1, 0,   1,-1, 1,  0,-1, 0,
             1,-1, 1,  0,-1, 0,  -1,-1, 1,  0,-1, 0,  -1,-1,-1,  0,-1, 0,
            -1, 1,-1,  0, 1, 0,   1, 1, 1,  0, 1, 0,   1, 1,-1,  0, 1, 0,
             1, 1, 1,  0, 1, 0,  -1, 1,-1,  0, 1, 0,  -1, 1, 1,  0, 1, 0
        };

        GLuint vao = 0;
        GLuint cubeVbo = 0;
        GLuint instanceVbo = 0;
        GLuint fullscreenVao = 0;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &cubeVbo);
        glGenBuffers(1, &instanceVbo);
        glGenVertexArrays(1, &fullscreenVao);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, cubeVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));

        glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
        glBufferData(GL_ARRAY_BUFFER, kGridW * kGridH * sizeof(Instance), nullptr, GL_STREAM_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), nullptr);
        glVertexAttribDivisor(2, 1);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Instance), reinterpret_cast<void*>(sizeof(glm::vec4)));
        glVertexAttribDivisor(3, 1);
        glBindVertexArray(0);

        RenderTarget sceneTarget;
        ensureRenderTarget(sceneTarget, state.framebufferW, state.framebufferH);

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

                if (!previousGray.empty()) {
                    cv::calcOpticalFlowFarneback(previousGray, gray, flow, 0.5, 3, 9, 2, 5, 1.1, 0);
                } else {
                    flow = cv::Mat::zeros(gray.size(), CV_32FC2);
                }
                gray.copyTo(previousGray);

                if (state.depthEnabled && depthEstimator.available()) {
                    if (depthNearness.empty() || (capturedFrameIndex % kDepthInferenceInterval) == 0) {
                        cv::Mat inferred = depthEstimator.inferNearness(frame, cv::Size(kGridW, kGridH));
                        if (!inferred.empty()) depthNearness = inferred;
                    }
                } else {
                    depthNearness = luminanceFallbackNearness(small);
                }

                if (depthNearness.empty()) {
                    depthNearness = luminanceFallbackNearness(small);
                }

                instances = frameToVoxels(small, flow, depthNearness);
                glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
                glBufferSubData(
                    GL_ARRAY_BUFFER,
                    0,
                    static_cast<GLsizeiptr>(instances.size() * sizeof(Instance)),
                    instances.data());

                ++capturedFrameIndex;
            }

            ensureRenderTarget(sceneTarget, state.framebufferW, state.framebufferH);

            const float cp = std::cos(state.pitch);
            const glm::vec3 cameraPosition(
                state.distance * cp * std::sin(state.yaw),
                state.distance * std::sin(state.pitch),
                state.distance * cp * std::cos(state.yaw));

            const glm::mat4 view = glm::lookAt(cameraPosition, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            const float aspect = static_cast<float>(state.framebufferW) / static_cast<float>(std::max(state.framebufferH, 1));
            const float renderFov = state.flyVisionEnabled ? kFlyPreviewFovDegrees : kResearcherFovDegrees;
            const glm::mat4 projection = glm::perspective(renderFov * kPi / 180.0f, aspect, 0.05f, 100.0f);

            // Pass 1: ordinary OpenGL voxel scene into a colour/depth framebuffer.
            glBindFramebuffer(GL_FRAMEBUFFER, sceneTarget.framebuffer);
            glViewport(0, 0, sceneTarget.width, sceneTarget.height);
            glEnable(GL_DEPTH_TEST);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUseProgram(sceneProgram);
            glUniformMatrix4fv(glGetUniformLocation(sceneProgram, "uView"), 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(glGetUniformLocation(sceneProgram, "uProjection"), 1, GL_FALSE, glm::value_ptr(projection));
            glBindVertexArray(vao);
            glDrawArraysInstanced(GL_TRIANGLES, 0, 36, static_cast<GLsizei>(instances.size()));
            glBindVertexArray(0);

            // Pass 2: researcher view is a transparent copy; fly mode performs the
            // compound-eye post-process over exactly the same rendered world.
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, state.framebufferW, state.framebufferH);
            glDisable(GL_DEPTH_TEST);
            glClear(GL_COLOR_BUFFER_BIT);

            glUseProgram(perceptionProgram);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, sceneTarget.colourTexture);
            glUniform1i(glGetUniformLocation(perceptionProgram, "uScene"), 0);
            glUniform2f(
                glGetUniformLocation(perceptionProgram, "uResolution"),
                static_cast<float>(state.framebufferW),
                static_cast<float>(state.framebufferH));
            glUniform1i(glGetUniformLocation(perceptionProgram, "uFlyVision"), state.flyVisionEnabled ? 1 : 0);
            glBindVertexArray(fullscreenVao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);
            glEnable(GL_DEPTH_TEST);

            glfwSwapBuffers(window);

            titleAccumulator += dt;
            ++titleFrames;
            if (titleAccumulator >= 0.5) {
                const double fps = static_cast<double>(titleFrames) / titleAccumulator;
                const bool realDepth = state.depthEnabled && depthEstimator.available();
                std::string depthLabel = realDepth ? "DEPTH" : "FALLBACK";
                if (realDepth) {
                    depthLabel += " " + std::to_string(static_cast<int>(depthEstimator.lastInferenceMs())) + "ms";
                }

                const std::string perceptionLabel = state.flyVisionEnabled ? "FLY" : "RESEARCHER";
                const std::string title =
                    "FlyVoxelisedReality | M2 " + perceptionLabel + " | " + depthLabel + " | " +
                    std::to_string(static_cast<int>(fps)) + " FPS | " +
                    std::to_string(instances.size()) + " voxels" +
                    (state.paused ? " | PAUSED" : "");
                glfwSetWindowTitle(window, title.c_str());
                titleAccumulator = 0.0;
                titleFrames = 0;
            }
        }

        if (capture.isOpened()) capture.release();
        destroyRenderTarget(sceneTarget);
        glDeleteVertexArrays(1, &fullscreenVao);
        glDeleteBuffers(1, &instanceVbo);
        glDeleteBuffers(1, &cubeVbo);
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(perceptionProgram);
        glDeleteProgram(sceneProgram);

        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        glfwTerminate();
        return 1;
    }
}
