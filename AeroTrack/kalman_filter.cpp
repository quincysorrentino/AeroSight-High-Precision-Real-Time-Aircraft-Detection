#include "tracker.h"

namespace
{
    constexpr float kProcessNoise = 3e-3f;  // tunable: increase for more agility
    constexpr float kMeasNoiseBase = 5e-2f; // tunable: base measurement noise
    constexpr float kInitError = 0.1f;
    constexpr float kConfidenceScale = 0.1f;
    constexpr float kConfidenceOffset = 0.12f;
    constexpr float kMinAdaptiveNoise = 0.02f;
    constexpr float kMaxAdaptiveNoise = 0.15f;
    constexpr float kHalfScale = 0.5f;
}

GhostTracker::GhostTracker() : framesLost(0), initialized(false)
{
    // state: [x, y, vx, vy, w, h]
    kf = cv::KalmanFilter(6, 4, 0);
    kf.transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, 1, 0, 0, 0, // x = x + vx
                           0, 1, 0, 1, 0, 0,                          // y = y + vy
                           0, 0, 1, 0, 0, 0,                          // velocity stays same
                           0, 0, 0, 1, 0, 0,
                           0, 0, 0, 0, 1, 0, // width stays same
                           0, 0, 0, 0, 0, 1);

    // Map measurements [cx, cy, w, h] to state [x, y, vx, vy, w, h]
    kf.measurementMatrix = (cv::Mat_<float>(4, 6) << 1, 0, 0, 0, 0, 0,
                            0, 1, 0, 0, 0, 0,
                            0, 0, 0, 0, 1, 0,
                            0, 0, 0, 0, 0, 1);
    // Tuned noise parameters for smoother tracking
    setIdentity(kf.processNoiseCov, cv::Scalar::all(kProcessNoise));
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(kMeasNoiseBase));
    setIdentity(kf.errorCovPost, cv::Scalar::all(kInitError));
}

void GhostTracker::initialize(cv::Rect firstDetection)
{
    // Initialize state with first detection
    const float cx = firstDetection.x + firstDetection.width * kHalfScale;
    const float cy = firstDetection.y + firstDetection.height * kHalfScale;

    kf.statePost.at<float>(0) = cx;
    kf.statePost.at<float>(1) = cy;
    kf.statePost.at<float>(2) = 0.0f; // vx = 0
    kf.statePost.at<float>(3) = 0.0f; // vy = 0
    kf.statePost.at<float>(4) = firstDetection.width;
    kf.statePost.at<float>(5) = firstDetection.height;
    initialized = true;
    framesLost = 0;
}

cv::Rect GhostTracker::predict()
{
    if (!initialized)
        return {};

    const cv::Mat pred = kf.predict();
    const float cx = pred.at<float>(0);
    const float cy = pred.at<float>(1);
    const float w = pred.at<float>(4);
    const float h = pred.at<float>(5);
    const float half_w = w * kHalfScale;
    const float half_h = h * kHalfScale;

    return cv::Rect(cv::Point2f(cx - half_w, cy - half_h), cv::Size2f(w, h));
}

void GhostTracker::update(cv::Rect b, float confidence)
{
    // Calculate center coordinates
    const float cx = b.x + b.width * kHalfScale;
    const float cy = b.y + b.height * kHalfScale;
    const cv::Mat measure = (cv::Mat_<float>(4, 1) << cx, cy, b.width, b.height);

    // Cache predicted center to derive observed velocity
    const float preX = kf.statePre.at<float>(0);
    const float preY = kf.statePre.at<float>(1);

    // Adaptive measurement noise based on confidence
    const float adaptiveNoise = std::clamp(
        kMeasNoiseBase + (kConfidenceOffset - confidence * kConfidenceScale),
        kMinAdaptiveNoise,
        kMaxAdaptiveNoise);
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(adaptiveNoise));

    kf.correct(measure);

    // Inject measured velocity using residual to prevent drift
    kf.statePost.at<float>(2) = cx - preX; // vx
    kf.statePost.at<float>(3) = cy - preY; // vy
    framesLost = 0;
}

cv::Rect GhostTracker::getPrediction() const
{
    const float cx = kf.statePre.at<float>(0);
    const float cy = kf.statePre.at<float>(1);
    const float w = kf.statePre.at<float>(4);
    const float h = kf.statePre.at<float>(5);
    const float half_w = w * kHalfScale;
    const float half_h = h * kHalfScale;

    return cv::Rect(cx - half_w, cy - half_h, w, h);
}

void GhostTracker::incrementLostFrames() { ++framesLost; }
void GhostTracker::resetLostFrames() { framesLost = 0; }
int GhostTracker::getLostFrames() const { return framesLost; }
bool GhostTracker::isInitialized() const { return initialized; }