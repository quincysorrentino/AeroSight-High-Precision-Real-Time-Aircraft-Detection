#include "utils.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>

float calculateIOU(const cv::Rect &a, const cv::Rect &b)
{
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width, b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);

    const int intersect = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    const int unionArea = a.area() + b.area() - intersect;

    return unionArea > 0 ? static_cast<float>(intersect) / unionArea : 0.0f;
}

float calculateCenterDistance(const cv::Rect &a, const cv::Rect &b)
{
    const float dx = (a.x + a.width * 0.5f) - (b.x + b.width * 0.5f);
    const float dy = (a.y + a.height * 0.5f) - (b.y + b.height * 0.5f);
    return std::sqrt(dx * dx + dy * dy);
}

cv::Rect clampToFrame(const cv::Rect &box, int maxWidth, int maxHeight)
{
    const int clampedX = std::max(0, std::min(box.x, maxWidth - box.width));
    const int clampedY = std::max(0, std::min(box.y, maxHeight - box.height));
    const int clampedW = std::max(1, std::min(box.width, maxWidth - clampedX));
    const int clampedH = std::max(1, std::min(box.height, maxHeight - clampedY));
    return {clampedX, clampedY, clampedW, clampedH};
}

cv::Mat computeHSVHist(const cv::Mat &frame, const cv::Rect &box)
{
    const cv::Rect safeBox = clampToFrame(box, frame.cols, frame.rows);
    const cv::Mat roi = frame(safeBox);
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    constexpr int hBins = 16, sBins = 16;
    const int histSize[] = {hBins, sBins};
    constexpr float hRanges[] = {0, 180};
    constexpr float sRanges[] = {0, 256};
    const float *ranges[] = {hRanges, sRanges};
    const int channels[] = {0, 1};

    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    if (!hist.empty())
        cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

    return hist;
}

std::vector<Detection> applyNMS(const std::vector<Detection> &detections, float nmsThreshold)
{
    if (detections.empty())
        return {};

    std::vector<Detection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(),
              [](const Detection &a, const Detection &b)
              { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(sorted.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < sorted.size(); i++)
    {
        if (suppressed[i])
            continue;

        result.push_back(sorted[i]);

        for (size_t j = i + 1; j < sorted.size(); j++)
        {
            if (suppressed[j])
                continue;

            if (calculateIOU(sorted[i].box, sorted[j].box) > nmsThreshold)
                suppressed[j] = true;
        }
    }

    return result;
}

std::vector<std::string> getClassNames()
{
    return {
        "A10", "A400M", "AG600", "AH64", "AV8B", "B1", "B2", "B52", "Be200", "C130",
        "C17", "C2", "C5", "CH47", "CL415", "E2", "EF2000", "EMB314", "F117", "F14",
        "F15", "F16", "F18", "F22", "F35", "F4", "H6", "Il76", "J10", "J20",
        "JAS39", "JF17", "JH7", "KC135", "Ka52", "MQ9", "Mi24", "Mi8", "Mig29", "Mig31",
        "Mirage2000", "P3", "RQ4", "Rafale", "SR71", "Su24", "Su25", "Su34", "Su57", "TB2",
        "Tornado", "Tu160", "Tu22M", "Tu95", "U2", "UH60", "US2", "V22", "Vulcan", "Y20"};
}
