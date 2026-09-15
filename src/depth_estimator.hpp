#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>

class DepthEstimator {
public:
    explicit DepthEstimator(const std::string& modelPath);

    [[nodiscard]] bool available() const noexcept { return available_; }
    [[nodiscard]] const std::string& status() const noexcept { return status_; }
    [[nodiscard]] double lastInferenceMs() const noexcept { return lastInferenceMs_; }

    // Returns a CV_32F map in [0, 1], where 1 means nearer and 0 means farther.
    // MiDaS v2.1 produces relative inverse depth, so this is deliberately
    // relative geometry rather than metric distance in metres.
    cv::Mat inferNearness(const cv::Mat& bgrFrame, cv::Size outputSize);

private:
    cv::dnn::Net net_;
    bool available_ = false;
    std::string status_;
    double lastInferenceMs_ = 0.0;
    cv::Mat smoothedNearness_;
};
