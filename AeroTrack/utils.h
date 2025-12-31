#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include "detector.h"

// Geometry utilities
float calculateIOU(const cv::Rect &a, const cv::Rect &b);
float calculateCenterDistance(const cv::Rect &a, const cv::Rect &b);
cv::Rect clampToFrame(const cv::Rect &box, int maxWidth, int maxHeight);

// Appearance utilities
cv::Mat computeHSVHist(const cv::Mat &frame, const cv::Rect &box);

// Detection utilities
std::vector<Detection> applyNMS(const std::vector<Detection> &detections, float nmsThreshold = 0.5f);

// Aircraft class names
std::vector<std::string> getClassNames();
