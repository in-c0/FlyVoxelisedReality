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
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kGridW = 48;
constexpr int kGridH = 36;
constexpr float kVoxelSpacing = 0.16f;
constexpr float kPi = 3.14159265358979323846f;

struct Instance {
    glm::vec4 positionScale;
    glm::vec4 colorMotion;
};

struct AppState {
    int framebufferW = 1280;
    int framebufferH = 720;
    bool paused = false;
    bool dragging = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    float yaw = 0.0f;
    float pitch = 0.05f;
    float distance = 7.2f;
};

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

std::vector<Instance> frameToVoxels(const cv::Mat& bgrSmall, const cv::Mat& flow) {
    std::vector<Instance> instances;
    instances.reserve(static_cast<size_t>(kGridW * kGridH));

    for (int y = 0; y < kGridH; ++y) {
        const auto* pixels = bgrSmall.ptr<cv::Vec3b>(y);
        const cv::Point2f* flowRow = flow.empty() ? nullptr : flow.ptr<cv::Point2f>(y);

        for (int x = 0; x < kGridW; ++x) {
            const cv::Vec3b bgr = pixels[x];
            const glm::vec3 colour(
                static_cast<float>(bgr[2]) / 255.0f,
                static_cast<float>(bgr[1]) / 255.0f,
                static_cast<float>(bgr[0]) / 255.0f);

            const float luminance = glm::dot(colour, glm::vec3(0.2126f, 0.7152f, 0.0722f));
            float motion = 0.0f;
            if (flowRow) {
                const cv::Point2f f = flowRow[x];
                motion = std::clamp(std::sqrt(f.x * f.x + f.y * f.y) * 0.22f, 0.0f, 1.0f);
            }

            const float worldX = (static_cast<float>(x) - (kGridW - 1) * 0.5f) * kVoxelSpacing;
            const float worldY = ((kGridH - 1) * 0.5f - static_cast<float>(y)) * kVoxelSpacing;
            const float worldZ = (luminance - 0.5f) * 1.8f + motion * 0.20f;
            const float scale = 0.070f + motion * 0.025f;

            instances.push_back({
                glm::vec4(worldX, worldY, worldZ, scale),
                glm::vec4(colour, motion)
            });
        }
    }
    return instances;
}

} // namespace

int main() {
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

        GLFWwindow* window = glfwCreateWindow(1280, 720, "FlyVoxelisedReality | M0", nullptr, nullptr);
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

        const GLuint program = createProgram();

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
        glBufferData(GL_ARRAY_BUFFER, kGridW * kGridH * sizeof(Instance), nullptr, GL_STREAM_DRAW);
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
        std::vector<Instance> instances;

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

                instances = frameToVoxels(small, flow);
                glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
                glBufferSubData(
                    GL_ARRAY_BUFFER,
                    0,
                    static_cast<GLsizeiptr>(instances.size() * sizeof(Instance)),
                    instances.data());
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
                const std::string title =
                    "FlyVoxelisedReality | M0 | " + std::to_string(static_cast<int>(fps)) +
                    " FPS | " + std::to_string(instances.size()) + " voxels" +
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
