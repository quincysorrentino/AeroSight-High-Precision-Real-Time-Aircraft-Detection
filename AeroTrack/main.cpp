#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <algorithm>
#include <cmath>
#include <deque>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include "detector.h"

cv::Rect clampToFrame(const cv::Rect &box, int maxWidth, int maxHeight);
cv::Mat computeHSVHist(const cv::Mat &frame, const cv::Rect &box);

// Lightweight, Kalman-free track record with simple velocity extrapolation
struct Track
{
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
    std::map<int, int> classTally; // cumulative class duration to reduce label flipping

    Track(int trackId, cv::Rect box, int frame, int cls, const std::string &name, float conf)
        : id(trackId), lastBox(box), velocity(0.0f, 0.0f), lastSeenFrame(frame), classID(cls),
          className(name), lastConfidence(conf), bestConfidence(conf), bestClassID(cls), totalHits(1)
    {
        classHistory.push_back(cls);
        confHistory.push_back(conf);
        classTally[cls] = 1;
    }

    cv::Point2f center() const
    {
        return {lastBox.x + lastBox.width / 2.0f, lastBox.y + lastBox.height / 2.0f};
    }

    cv::Rect predictedBox(float damping = 0.9f) const
    {
        cv::Point2f newCenter = center() + velocity * damping;
        cv::Rect predicted(static_cast<int>(newCenter.x - lastBox.width / 2.0f),
                           static_cast<int>(newCenter.y - lastBox.height / 2.0f),
                           lastBox.width, lastBox.height);
        return predicted;
    }

    void applyPrediction(float damping, int maxWidth, int maxHeight)
    {
        cv::Rect predicted = clampToFrame(predictedBox(damping), maxWidth, maxHeight);
        lastBox = predicted;
        velocity *= damping;
    }

    void update(const cv::Rect &box, int frame, float conf, int cls, const std::string &name)
    {
        cv::Point2f currentCenter = center();
        cv::Point2f newCenter(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
        cv::Point2f measuredVelocity = newCenter - currentCenter;

        // Blend measured velocity with running estimate to avoid abrupt jumps
        velocity = 0.6f * measuredVelocity + 0.4f * velocity;

        lastBox = box;
        lastSeenFrame = frame;
        lastConfidence = conf;
        totalHits++;

        // Maintain rolling class/conf history (keep last 20 entries)
        const size_t MAX_HISTORY = 20;
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
    }

    int getFramesSinceLastSeen(int currentFrame) const
    {
        return currentFrame - lastSeenFrame;
    }

    void refreshKeypoints(int step = 10)
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
        {
            keypoints.push_back({static_cast<float>(lastBox.x + lastBox.width / 2.0f),
                                 static_cast<float>(lastBox.y + lastBox.height / 2.0f)});
        }
    }

    void updateWithFlow(const cv::Mat &prevGray, const cv::Mat &gray, int maxWidth, int maxHeight)
    {
        if (keypoints.empty())
            return;

        std::vector<cv::Point2f> nextPts;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(prevGray, gray, keypoints, nextPts, status, err, cv::Size(15, 15), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03));

        cv::Point2f sumDelta(0.0f, 0.0f);
        int good = 0;
        for (size_t i = 0; i < nextPts.size(); ++i)
        {
            if (status[i])
            {
                sumDelta += (nextPts[i] - keypoints[i]);
                good++;
            }
        }

        if (good > 0)
        {
            cv::Point2f avgDelta = sumDelta * (1.0f / good);
            velocity = 0.7f * avgDelta + 0.3f * velocity;

            cv::Rect shifted(lastBox.x + static_cast<int>(std::round(avgDelta.x)),
                             lastBox.y + static_cast<int>(std::round(avgDelta.y)),
                             lastBox.width, lastBox.height);
            lastBox = clampToFrame(shifted, maxWidth, maxHeight);
            keypoints.clear();
            for (size_t i = 0; i < nextPts.size(); ++i)
            {
                if (status[i])
                {
                    keypoints.push_back(nextPts[i]);
                }
            }
            if (keypoints.empty())
            {
                refreshKeypoints();
            }
        }
    }

    void updateAppearance(const cv::Mat &frame, const cv::Rect &box)
    {
        if (!frame.empty())
        {
            appearanceHist = computeHSVHist(frame, box);
        }
    }

    std::pair<int, float> majorityClass() const
    {
        // Prefer the class seen for the longest cumulative time to avoid flip-flopping
        if (classTally.empty())
            return {classID, lastConfidence};

        int bestCls = classID;
        int bestCount = 0;
        for (const auto &kv : classTally)
        {
            if (kv.second > bestCount)
            {
                bestCount = kv.second;
                bestCls = kv.first;
            }
        }

        float avgConf = 0.0f;
        for (float c : confHistory)
            avgConf += c;
        if (!confHistory.empty())
            avgConf /= static_cast<float>(confHistory.size());

        return {bestCls, avgConf};
    }

    bool markStaleIfOld(int currentFrame, int staleFrames)
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
};

// map the ID numbers to real names
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

// Helper function to calculate IOU
float calculateIOU(const cv::Rect &a, const cv::Rect &b)
{
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int intersect = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - intersect;

    return unionArea > 0 ? (float)intersect / unionArea : 0.0f;
}

// Helper function to calculate center-to-center distance
float calculateCenterDistance(const cv::Rect &a, const cv::Rect &b)
{
    float cx1 = a.x + a.width / 2.0f;
    float cy1 = a.y + a.height / 2.0f;
    float cx2 = b.x + b.width / 2.0f;
    float cy2 = b.y + b.height / 2.0f;

    return std::sqrt((cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2));
}

cv::Rect clampToFrame(const cv::Rect &box, int maxWidth, int maxHeight)
{
    int clampedX = std::max(0, std::min(box.x, maxWidth - box.width));
    int clampedY = std::max(0, std::min(box.y, maxHeight - box.height));
    int clampedW = std::max(1, std::min(box.width, maxWidth - clampedX));
    int clampedH = std::max(1, std::min(box.height, maxHeight - clampedY));
    return {clampedX, clampedY, clampedW, clampedH};
}

cv::Mat computeHSVHist(const cv::Mat &frame, const cv::Rect &box)
{
    cv::Rect safeBox = clampToFrame(box, frame.cols, frame.rows);
    cv::Mat roi = frame(safeBox);
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    int hBins = 16, sBins = 16;
    int histSize[] = {hBins, sBins};
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    const float *ranges[] = {hRanges, sRanges};
    int channels[] = {0, 1};

    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    if (!hist.empty())
    {
        cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    }
    return hist;
}

// Non-Maximum Suppression to filter duplicate/overlapping detections
std::vector<Detection> applyNMS(const std::vector<Detection> &detections, float nmsThreshold = 0.5f)
{
    if (detections.empty())
        return detections;

    // Sort by confidence descending
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

        // Suppress overlapping boxes
        for (size_t j = i + 1; j < sorted.size(); j++)
        {
            if (suppressed[j])
                continue;

            float iou = calculateIOU(sorted[i].box, sorted[j].box);
            if (iou > nmsThreshold)
            {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

int main()
{
    try
    {
        // initialize detector
        std::cout << "Loading model..." << std::endl;
        Detector detector(L"last.onnx");
        std::cout << "Model loaded successfully!" << std::endl;

        // --- VIDEO INITIALIZATION ---
        cv::VideoCapture cap("test.mp4");

        if (!cap.isOpened())
        {
            std::cerr << "ERROR: Could not open video source." << std::endl;
            return -1;
        }

        // Get video properties for output
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps_input = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        // Calculate output dimensions - maintain aspect ratio, fit to 1920x1080
        int output_width = 1920;
        int output_height = 1080;
        float scale = std::min((float)output_width / frame_width, (float)output_height / frame_height);
        int scaled_width = static_cast<int>(frame_width * scale);
        int scaled_height = static_cast<int>(frame_height * scale);
        int offset_x = (output_width - scaled_width) / 2;
        int offset_y = (output_height - scaled_height) / 2;

        // Initialize video writer with standard landscape dimensions
        cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                               fps_input, cv::Size(output_width, output_height));

        if (!writer.isOpened())
        {
            std::cerr << "ERROR: Could not open video writer." << std::endl;
            return -1;
        }

        std::cout << "Input video: " << frame_width << "x" << frame_height
                  << " @ " << fps_input << " FPS, " << total_frames << " frames" << std::endl;
        std::cout << "Output video: " << output_width << "x" << output_height << std::endl;
        std::cout << "Writing to output.mp4..." << std::endl;

        std::vector<std::string> classNames = getClassNames();
        cv::Mat frame, resized_frame, output_frame;
        cv::Mat prev_gray, gray;
        int frame_count = 0;

        // Simplified tracking parameters
        std::vector<Track> activeTracks;
        int nextTrackID = 1;
        const int MAX_FRAMES_LOST = 450;     // Keep track alive for a long duration
        const float MATCH_DISTANCE = 450.0f; // Base gate
        const float MATCH_IOU = 0.05f;       // Slightly higher IOU for sanity
        const float MERGE_DISTANCE = 80.0f;  // Conservative merge radius to avoid churn
        const float MERGE_IOU = 0.65f;
        const float CONFIDENCE_THRESHOLD = 0.12f; // Accept low confidence to avoid dropping tracks
        const float PREDICTION_DAMPING = 0.90f;   // Slightly stronger damping
        const int CLASS_STALE_FRAMES = 90;        // After this many frames unseen, clear class label

        std::cout << "Processing video (no preview window)..." << std::endl;

        // --- THE VIDEO LOOP ---
        while (true)
        {
            auto start = std::chrono::high_resolution_clock::now();

            cap >> frame;
            if (frame.empty())
                break;

            frame_count++;
            std::cout << "Processing frame " << frame_count << "/" << total_frames << "\r" << std::flush;

            // Resize frame to fit output dimensions while maintaining aspect ratio
            cv::resize(frame, resized_frame, cv::Size(scaled_width, scaled_height));

            // Prepare grayscale for optical flow
            cv::cvtColor(resized_frame, gray, cv::COLOR_BGR2GRAY);

            // Create black canvas and place resized frame in center
            output_frame = cv::Mat::zeros(output_height, output_width, frame.type());
            resized_frame.copyTo(output_frame(cv::Rect(offset_x, offset_y, scaled_width, scaled_height)));

            // === STEP 0: Optical flow keep-alive ===
            if (!prev_gray.empty())
            {
                for (auto &track : activeTracks)
                {
                    track.updateWithFlow(prev_gray, gray, scaled_width, scaled_height);
                }
            }

            // === STEP 1: Run YOLO detection ===
            std::vector<Detection> results = detector.run(resized_frame);

            // Filter by confidence threshold (accept low confidence)
            std::vector<Detection> validDetections;
            for (const auto &det : results)
            {
                if (det.confidence >= CONFIDENCE_THRESHOLD)
                {
                    validDetections.push_back(det);
                }
            }

            // === STEP 2: Prepare detection appearance hists ===
            std::vector<cv::Mat> detectionHists;
            detectionHists.reserve(validDetections.size());
            for (const auto &det : validDetections)
            {
                detectionHists.push_back(computeHSVHist(resized_frame, det.box));
            }

            // === STEP 3: Match detections to existing tracks ===
            std::vector<bool> detectionMatched(validDetections.size(), false);
            std::vector<int> trackMatchedTo(activeTracks.size(), -1);

            std::vector<cv::Rect> predictedBoxes;
            predictedBoxes.reserve(activeTracks.size());
            for (const auto &track : activeTracks)
            {
                predictedBoxes.push_back(clampToFrame(track.predictedBox(PREDICTION_DAMPING), scaled_width, scaled_height));
            }

            // Score each track-detection pair
            std::vector<std::tuple<float, int, int>> matchScores; // (score, trackIdx, detIdx)

            // DEBUG: Print matching attempts
            if (frame_count <= 20 && !activeTracks.empty() && !validDetections.empty())
            {
                std::cout << "\n[DEBUG] Frame " << frame_count << ": " << activeTracks.size()
                          << " tracks, " << validDetections.size() << " detections" << std::endl;
            }

            for (size_t t = 0; t < activeTracks.size(); t++)
            {
                for (size_t d = 0; d < validDetections.size(); d++)
                {
                    float distance = calculateCenterDistance(predictedBoxes[t], validDetections[d].box);
                    float iou = calculateIOU(predictedBoxes[t], validDetections[d].box);

                    // Appearance similarity (correlation in HSV)
                    double appearanceSim = 0.0;
                    if (!activeTracks[t].appearanceHist.empty() && !detectionHists[d].empty())
                    {
                        appearanceSim = cv::compareHist(activeTracks[t].appearanceHist, detectionHists[d], cv::HISTCMP_CORREL);
                    }

                    // Favor consistent classes but still allow cross-class continuation
                    float classBonus = (activeTracks[t].bestClassID == validDetections[d].classID) ? 1.0f : 0.1f;
                    float confidenceBoost = validDetections[d].confidence * 0.5f;

                    float velMag = std::sqrt(activeTracks[t].velocity.x * activeTracks[t].velocity.x +
                                             activeTracks[t].velocity.y * activeTracks[t].velocity.y);
                    float dynamicGate = MATCH_DISTANCE + std::min(200.0f, velMag * 5.0f);

                    if (frame_count <= 20 && (distance < MATCH_DISTANCE || iou > MATCH_IOU))
                    {
                        std::cout << "  Track ID" << activeTracks[t].id << " <-> Det" << d
                                  << ": dist=" << (int)distance << "px, iou=" << iou
                                  << ", class_bonus=" << classBonus << std::endl;
                    }

                    if (distance < dynamicGate || iou > MATCH_IOU)
                    {
                        float score = (iou * 3.0f) - (distance / 900.0f) + classBonus + confidenceBoost + static_cast<float>(appearanceSim * 1.2);
                        matchScores.push_back(std::make_tuple(score, t, d));
                    }
                }
            }

            // Sort by score descending (best matches first)
            std::sort(matchScores.begin(), matchScores.end(),
                      [](const auto &a, const auto &b)
                      { return std::get<0>(a) > std::get<0>(b); });

            // Assign best matches
            for (const auto &score : matchScores)
            {
                int trackIdx = std::get<1>(score);
                int detIdx = std::get<2>(score);

                if (trackMatchedTo[trackIdx] == -1 && !detectionMatched[detIdx])
                {
                    trackMatchedTo[trackIdx] = detIdx;
                    detectionMatched[detIdx] = true;
                }
            }

            // === STEP 4: Update matched tracks ===
            for (size_t t = 0; t < activeTracks.size(); t++)
            {
                if (trackMatchedTo[t] >= 0)
                {
                    const Detection &det = validDetections[trackMatchedTo[t]];
                    std::string aircraftName = (det.classID >= 0 && det.classID < classNames.size())
                                                   ? classNames[det.classID]
                                                   : "Unknown";
                    activeTracks[t].update(det.box, frame_count, det.confidence, det.classID, aircraftName);
                    activeTracks[t].updateAppearance(resized_frame, det.box);
                    activeTracks[t].refreshKeypoints();
                }
            }

            // === STEP 4b: Propagate unmatched tracks forward ===
            for (size_t t = 0; t < activeTracks.size(); t++)
            {
                if (trackMatchedTo[t] == -1)
                {
                    activeTracks[t].applyPrediction(PREDICTION_DAMPING, scaled_width, scaled_height);
                    activeTracks[t].markStaleIfOld(frame_count, CLASS_STALE_FRAMES);
                }
            }

            // === STEP 5: Create new tracks for unmatched detections ===
            for (size_t d = 0; d < validDetections.size(); d++)
            {
                if (!detectionMatched[d])
                {
                    const Detection &det = validDetections[d];
                    std::string aircraftName = (det.classID >= 0 && det.classID < classNames.size())
                                                   ? classNames[det.classID]
                                                   : "Unknown";

                    activeTracks.emplace_back(nextTrackID++, det.box, frame_count,
                                              det.classID, aircraftName, det.confidence);
                    activeTracks.back().updateAppearance(resized_frame, det.box);
                    activeTracks.back().refreshKeypoints();

                    std::cout << "\n[NEW TRACK] ID" << activeTracks.back().id << ": "
                              << aircraftName << " @ " << (int)(det.confidence * 100) << "%" << std::endl;
                }
            }

            // === STEP 5: Merge overlapping tracks (more conservative) ===
            for (size_t i = 0; i < activeTracks.size(); i++)
            {
                for (size_t j = i + 1; j < activeTracks.size(); j++)
                {
                    float distance = calculateCenterDistance(activeTracks[i].lastBox, activeTracks[j].lastBox);
                    float iou = calculateIOU(activeTracks[i].lastBox, activeTracks[j].lastBox);

                    if (distance < MERGE_DISTANCE && iou > MERGE_IOU)
                    {
                        size_t keep = (activeTracks[i].bestConfidence >= activeTracks[j].bestConfidence) ? i : j;
                        size_t drop = (keep == i) ? j : i;
                        std::cout << "\n[MERGE] Removing ID" << activeTracks[drop].id
                                  << " (duplicate of ID" << activeTracks[keep].id << ")" << std::endl;
                        activeTracks.erase(activeTracks.begin() + drop);
                        if (drop < i)
                            i--;
                        j = i + 1;
                    }
                }
            }

            // === STEP 6: Remove old tracks ===
            activeTracks.erase(
                std::remove_if(activeTracks.begin(), activeTracks.end(),
                               [frame_count, MAX_FRAMES_LOST](const Track &t)
                               {
                                   bool shouldRemove = t.getFramesSinceLastSeen(frame_count) > MAX_FRAMES_LOST;
                                   if (shouldRemove)
                                   {
                                       std::cout << "\n[TRACK DELETED] ID" << t.id << ": " << t.className
                                                 << " (not seen for " << t.getFramesSinceLastSeen(frame_count) << " frames)" << std::endl;
                                   }
                                   return shouldRemove;
                               }),
                activeTracks.end());

            // === STEP 7: Draw all active tracks ===
            for (const auto &track : activeTracks)
            {
                cv::Rect adjusted_box(track.lastBox.x + offset_x, track.lastBox.y + offset_y,
                                      track.lastBox.width, track.lastBox.height);

                // Draw box
                cv::rectangle(output_frame, adjusted_box, cv::Scalar(0, 255, 0), 2);

                // Draw label based on majority class and averaged confidence
                auto maj = track.majorityClass();
                int labelCls = (maj.first >= 0 && maj.first < (int)classNames.size()) ? maj.first : track.classID;
                std::string labelName = (labelCls >= 0 && labelCls < (int)classNames.size()) ? classNames[labelCls] : track.className;
                float displayConf = std::max({track.lastConfidence, track.bestConfidence, maj.second});
                std::string label = "ID" + std::to_string(track.id) + ": " + labelName +
                                    " " + std::to_string((int)(displayConf * 100)) + "%";

                int baseLine;
                cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
                cv::rectangle(output_frame,
                              cv::Point(adjusted_box.x, adjusted_box.y - textSize.height - 8),
                              cv::Point(adjusted_box.x + textSize.width, adjusted_box.y),
                              cv::Scalar(0, 255, 0), -1);

                cv::putText(output_frame, label, {adjusted_box.x, adjusted_box.y - 5},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            }

            // === Display active track count ===
            std::string trackInfo = "Active Tracks: " + std::to_string(activeTracks.size());
            cv::putText(output_frame, trackInfo, {10, output_height - 20},
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);

            // Write frame to output video
            writer.write(output_frame);

            // carry gray frame forward for optical flow
            prev_gray = gray.clone();
        }

        std::cout << "\nProcessing complete! Output saved to output.mp4" << std::endl;
        std::cout << "Total frames processed: " << frame_count << std::endl;

        cap.release();
        writer.release();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
}
