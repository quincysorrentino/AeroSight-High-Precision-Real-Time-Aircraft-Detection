#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <map>
#include <deque>
#include "kalman_filter.h"

enum class TrackState
{
    Tracked,
    Lost,
    Removed
};

class Track
{
public:
    int id;
    cv::Rect lastBox;
    cv::Point2f velocity;
    int lastSeenFrame;
    int classID;
    std::string className;
    float lastConfidence;
    float bestConfidence;
    int bestClassID;
    int totalHits;
    cv::Mat appearanceHist;
    std::vector<cv::Point2f> keypoints;
    std::deque<int> classHistory;
    std::deque<float> confHistory;
    std::map<int, int> classTally;
    std::deque<cv::Point2f> trail;
    int lostFrames = 0;
    TrackState state = TrackState::Tracked;
    cv::Rect lastPredBox;
    GhostTracker motion;

    Track(int trackId, cv::Rect box, int frame, int cls, const std::string &name, float conf);

    cv::Point2f center() const;
    cv::Rect predictedBox(float damping = 0.9f) const;
    cv::Rect predictMotion(float damping, int maxWidth, int maxHeight);
    void applyPrediction(float damping, int maxWidth, int maxHeight);
    void update(const cv::Rect &box, int frame, float conf, int cls, const std::string &name);
    int getFramesSinceLastSeen(int currentFrame) const;
    void refreshKeypoints(int step = 10);
    void pushTrail();
    void updateWithFlow(const cv::Mat &prevGray, const cv::Mat &gray, int maxWidth, int maxHeight);
    void updateAppearance(const cv::Mat &frame, const cv::Rect &box);
    std::pair<int, float> majorityClass() const;
    bool markStaleIfOld(int currentFrame, int staleFrames);
};
