#include "track.h"
#include "utils.h"
#include <opencv2/video.hpp>
#include <algorithm>
#include <numeric>

namespace
{
    constexpr size_t MAX_HISTORY = 20;
    constexpr size_t MAX_TRAIL = 20;
    constexpr float VELOCITY_BLEND_NEW = 0.6f;
    constexpr float VELOCITY_BLEND_OLD = 0.4f;
    constexpr float FLOW_VELOCITY_NEW = 0.7f;
    constexpr float FLOW_VELOCITY_OLD = 0.3f;
}

Track::Track(int trackId, cv::Rect box, int frame, int cls, const std::string &name, float conf)
    : id(trackId), lastBox(box), velocity(0.0f, 0.0f), lastSeenFrame(frame), classID(cls),
      className(name), lastConfidence(conf), bestConfidence(conf), bestClassID(cls), totalHits(1)
{
    classHistory.push_back(cls);
    confHistory.push_back(conf);
    classTally[cls] = 1;
    trail.push_back(center());
    lastPredBox = box;
}

cv::Point2f Track::center() const
{
    return {lastBox.x + lastBox.width * 0.5f, lastBox.y + lastBox.height * 0.5f};
}

cv::Rect Track::predictedBox(float damping) const
{
    const cv::Point2f c = center();
    const cv::Point2f newCenter = c + velocity * damping;
    return cv::Rect(static_cast<int>(newCenter.x - lastBox.width * 0.5f),
                    static_cast<int>(newCenter.y - lastBox.height * 0.5f),
                    lastBox.width, lastBox.height);
}

cv::Rect Track::predictMotion(float damping, int maxWidth, int maxHeight)
{
    const cv::Rect pred = motion.isInitialized() ? motion.getPrediction() : predictedBox(damping);
    return clampToFrame(pred, maxWidth, maxHeight);
}

void Track::applyPrediction(float damping, int maxWidth, int maxHeight)
{
    const cv::Rect pred = motion.isInitialized() ? motion.getPrediction() : predictedBox(damping);
    lastBox = clampToFrame(pred, maxWidth, maxHeight);
    velocity *= damping;
}

void Track::update(const cv::Rect &box, int frame, float conf, int cls, const std::string &name)
{
    const cv::Point2f newCenter(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
    velocity = (newCenter - center()) * VELOCITY_BLEND_NEW + velocity * VELOCITY_BLEND_OLD;

    lastBox = box;
    lastSeenFrame = frame;
    state = TrackState::Tracked;
    lostFrames = 0;
    lastConfidence = conf;
    totalHits++;

    classHistory.push_back(cls);
    confHistory.push_back(conf);
    if (classHistory.size() > MAX_HISTORY)
        classHistory.pop_front();
    if (confHistory.size() > MAX_HISTORY)
        confHistory.pop_front();

    classTally[cls]++;

    if (conf >= bestConfidence)
    {
        bestConfidence = conf;
        bestClassID = cls;
        className = name;
        classID = cls;
    }

    if (!motion.isInitialized())
        motion.initialize(box);
    else
        motion.update(box, conf);
}

int Track::getFramesSinceLastSeen(int currentFrame) const
{
    return currentFrame - lastSeenFrame;
}

void Track::refreshKeypoints(int step)
{
    keypoints.clear();
    for (int y = lastBox.y + step; y < lastBox.y + lastBox.height - step; y += step)
    {
        for (int x = lastBox.x + step; x < lastBox.x + lastBox.width - step; x += step)
        {
            keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y));
        }
    }
    if (keypoints.empty())
        keypoints.push_back(center());
}

void Track::pushTrail()
{
    trail.push_back(center());
    if (trail.size() > MAX_TRAIL)
        trail.pop_front();
}

void Track::updateWithFlow(const cv::Mat &prevGray, const cv::Mat &gray, int maxWidth, int maxHeight)
{
    if (keypoints.empty())
        return;

    std::vector<cv::Point2f> nextPts;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(prevGray, gray, keypoints, nextPts, status, err,
                             cv::Size(15, 15), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03));

    cv::Point2f sumDelta(0.0f, 0.0f);
    std::vector<cv::Point2f> validNextPts;
    validNextPts.reserve(nextPts.size());

    for (size_t i = 0; i < nextPts.size(); ++i)
    {
        if (status[i])
        {
            sumDelta += (nextPts[i] - keypoints[i]);
            validNextPts.push_back(nextPts[i]);
        }
    }

    if (!validNextPts.empty())
    {
        const cv::Point2f avgDelta = sumDelta / static_cast<float>(validNextPts.size());
        velocity = avgDelta * FLOW_VELOCITY_NEW + velocity * FLOW_VELOCITY_OLD;

        lastBox.x += static_cast<int>(std::round(avgDelta.x));
        lastBox.y += static_cast<int>(std::round(avgDelta.y));
        lastBox = clampToFrame(lastBox, maxWidth, maxHeight);

        keypoints = std::move(validNextPts);
        if (keypoints.empty())
            refreshKeypoints();
    }
}

void Track::updateAppearance(const cv::Mat &frame, const cv::Rect &box)
{
    if (!frame.empty())
        appearanceHist = computeHSVHist(frame, box);
}

std::pair<int, float> Track::majorityClass() const
{
    if (classTally.empty())
        return {classID, lastConfidence};

    const auto maxElem = std::max_element(classTally.begin(), classTally.end(),
                                          [](const auto &a, const auto &b)
                                          { return a.second < b.second; });

    const float avgConf = confHistory.empty() ? 0.0f : std::accumulate(confHistory.begin(), confHistory.end(), 0.0f) / confHistory.size();

    return {maxElem->first, avgConf};
}

bool Track::markStaleIfOld(int currentFrame, int staleFrames)
{
    if (getFramesSinceLastSeen(currentFrame) > staleFrames)
    {
        classID = -1;
        className = "Unknown";
        bestConfidence = 0.0f;
        lastConfidence = 0.0f;
        classHistory.clear();
        confHistory.clear();
        classTally.clear();
        return true;
    }
    return false;
}
