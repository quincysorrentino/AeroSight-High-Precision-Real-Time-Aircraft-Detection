#include "tracker.h"

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

    setIdentity(kf.measurementMatrix);
    // Tuned noise parameters for smoother tracking
    setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-3));     // Increased from 1e-4
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(5e-2)); // Reduced from 1e-1
    setIdentity(kf.errorCovPost, cv::Scalar::all(0.1));
}

void GhostTracker::initialize(cv::Rect firstDetection)
{
    // Properly initialize state with first detection
    kf.statePost.at<float>(0) = firstDetection.x + firstDetection.width / 2.0f;  // center x
    kf.statePost.at<float>(1) = firstDetection.y + firstDetection.height / 2.0f; // center y
    kf.statePost.at<float>(2) = 0;                                               // vx = 0
    kf.statePost.at<float>(3) = 0;                                               // vy = 0
    kf.statePost.at<float>(4) = firstDetection.width;
    kf.statePost.at<float>(5) = firstDetection.height;
    initialized = true;
    framesLost = 0;
}

void GhostTracker::predict()
{
    if (initialized)
    {
        kf.predict();
    }
}

void GhostTracker::update(cv::Rect b, float confidence)
{
    // Update with center coordinates for better tracking
    float cx = b.x + b.width / 2.0f;
    float cy = b.y + b.height / 2.0f;
    cv::Mat measure = (cv::Mat_<float>(4, 1) << cx, cy, b.width, b.height);

    // Adaptive measurement noise based on confidence
    // High confidence (0.9) -> low noise (2e-2)
    // Low confidence (0.7) -> high noise (1e-1)
    float adaptiveNoise = 0.15f - (confidence * 0.13f);              // Maps 0.7->0.059, 0.9->0.033
    adaptiveNoise = std::max(0.02f, std::min(0.15f, adaptiveNoise)); // Clamp between 0.02 and 0.15

    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(adaptiveNoise));

    kf.correct(measure);
    framesLost = 0;
}

cv::Rect GhostTracker::getPrediction() const
{
    float cx = kf.statePost.at<float>(0);
    float cy = kf.statePost.at<float>(1);
    float w = kf.statePost.at<float>(4);
    float h = kf.statePost.at<float>(5);
    // Convert from center coordinates back to top-left
    return cv::Rect(cx - w / 2.0f, cy - h / 2.0f, w, h);
}

void GhostTracker::incrementLostFrames() { framesLost++; }
void GhostTracker::resetLostFrames() { framesLost = 0; }
int GhostTracker::getLostFrames() const { return framesLost; }
bool GhostTracker::isInitialized() const { return initialized; }