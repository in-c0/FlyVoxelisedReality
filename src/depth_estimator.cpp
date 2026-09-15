#include "depth_estimator.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kInputSize = 256;

float percentile(std::vector<float> values, float p) {
    if (values.empty()) return 0.0f;
    p = std::clamp(p, 0.0f, 1.0f);
    const size_t index = static_cast<size_t>(p * static_cast<float>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

cv::Mat predictionTo2D(const cv::Mat& prediction) {
    if (prediction.empty()) return {};

    if (prediction.dims == 4) {
        const int h = prediction.size[2];
        const int w = prediction.size[3];
        return cv::Mat(h, w, CV_32F, const_cast<float*>(prediction.ptr<float>())).clone();
    }

    if (prediction.dims == 3) {
        const int h = prediction.size[1];
        const int w = prediction.size[2];
        return cv::Mat(h, w, CV_32F, const_cast<float*>(prediction.ptr<float>())).clone();
    }

    if (prediction.dims == 2) {
        cv::Mat result;
        prediction.convertTo(result, CV_32F);
        return result;
    }

    throw std::runtime_error("Unexpected MiDaS output rank: " + std::to_string(prediction.dims));
}

} // namespace

DepthEstimator::DepthEstimator(const std::string& modelPath) {
    if (!std::filesystem::exists(modelPath)) {
        status_ = "model missing: " + modelPath;
        return;
    }

    try {
        net_ = cv::dnn::readNet(modelPath);
        if (net_.empty()) {
            status_ = "OpenCV DNN returned an empty network";
            return;
        }

        // CPU is the most portable v0 target. We can switch to a GPU backend
        // later without changing the renderer/data interface.
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        available_ = true;
        status_ = "MiDaS v2.1 small ready";
    } catch (const cv::Exception& e) {
        status_ = std::string("failed to load depth model: ") + e.what();
    }
}

cv::Mat DepthEstimator::inferNearness(const cv::Mat& bgrFrame, cv::Size outputSize) {
    if (!available_ || bgrFrame.empty()) return {};

    const auto start = std::chrono::steady_clock::now();

    try {
        cv::Mat resized;
        cv::resize(bgrFrame, resized, cv::Size(kInputSize, kInputSize), 0.0, 0.0, cv::INTER_AREA);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels;
        cv::split(rgb, channels);
        channels[0] = (channels[0] - 0.485f) / 0.229f;
        channels[1] = (channels[1] - 0.456f) / 0.224f;
        channels[2] = (channels[2] - 0.406f) / 0.225f;
        cv::merge(channels, rgb);

        cv::Mat blob = cv::dnn::blobFromImage(rgb);
        net_.setInput(blob);
        cv::Mat prediction = predictionTo2D(net_.forward());

        std::vector<float> finiteValues;
        finiteValues.reserve(prediction.total());
        for (int y = 0; y < prediction.rows; ++y) {
            const float* row = prediction.ptr<float>(y);
            for (int x = 0; x < prediction.cols; ++x) {
                const float v = row[x];
                if (std::isfinite(v)) finiteValues.push_back(v);
            }
        }

        if (finiteValues.size() < 16) {
            status_ = "depth inference produced too few finite values";
            return {};
        }

        const float lo = percentile(finiteValues, 0.02f);
        const float hi = percentile(finiteValues, 0.98f);
        const float range = std::max(hi - lo, 1e-6f);

        cv::Mat nearness(prediction.size(), CV_32F);
        for (int y = 0; y < prediction.rows; ++y) {
            const float* src = prediction.ptr<float>(y);
            float* dst = nearness.ptr<float>(y);
            for (int x = 0; x < prediction.cols; ++x) {
                const float v = std::isfinite(src[x]) ? src[x] : lo;
                dst[x] = std::clamp((v - lo) / range, 0.0f, 1.0f);
            }
        }

        cv::Mat output;
        cv::resize(nearness, output, outputSize, 0.0, 0.0, cv::INTER_CUBIC);

        // MiDaS relative depth changes slightly frame to frame. Temporal smoothing
        // makes the voxel surface readable without hiding large scene motion.
        if (smoothedNearness_.empty() || smoothedNearness_.size() != output.size()) {
            smoothedNearness_ = output.clone();
        } else {
            cv::addWeighted(output, 0.40, smoothedNearness_, 0.60, 0.0, smoothedNearness_);
        }

        const auto end = std::chrono::steady_clock::now();
        lastInferenceMs_ = std::chrono::duration<double, std::milli>(end - start).count();
        status_ = "MiDaS relative depth active";
        return smoothedNearness_.clone();
    } catch (const cv::Exception& e) {
        status_ = std::string("depth inference failed: ") + e.what();
        available_ = false;
        return {};
    }
}
