#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include "detector.h"
#include "track.h"
#include "utils.h"
#include "hungarian.h"

namespace
{
    // Tracking parameters
    constexpr int MAX_FRAMES_LOST = 200;
    constexpr float DIST_GATE_BASE = 450.0f;
    constexpr float IOU_GATE = 0.05f;
    constexpr float MERGE_DISTANCE = 80.0f;
    constexpr float MERGE_IOU = 0.65f;
    constexpr float HIGH_CONF = 0.35f;
    constexpr float LOW_CONF = 0.12f;
    constexpr float PREDICTION_DAMPING = 0.90f;
    constexpr int CLASS_STALE_FRAMES = 90;
    constexpr float MAX_ASSIGN_COST = 2.5f;
    constexpr float COSINE_PENALTY = 0.25f;
    constexpr float APPEARANCE_GATE = 0.05f;

    // Video output parameters
    constexpr int OUTPUT_WIDTH = 1920;
    constexpr int OUTPUT_HEIGHT = 1080;
}

int main(int argc, char **argv)
{
    try
    {
        // CLI: --kalman-test enables a Kalman-only window; optional "--kalman-test=start,duration" overrides timings (seconds)
        bool kalmanTest = false;
        double kalmanHoldStart = 3.0;
        double kalmanHoldDuration = 3.0;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            const std::string flag = "--kalman-test";
            if (arg == flag)
            {
                kalmanTest = true;
            }
            else if (arg.rfind(flag + "=", 0) == 0)
            {
                kalmanTest = true;
                std::string payload = arg.substr(flag.size() + 1);
                size_t comma = payload.find(',');
                if (comma != std::string::npos)
                {
                    try
                    {
                        kalmanHoldStart = std::stod(payload.substr(0, comma));
                        kalmanHoldDuration = std::stod(payload.substr(comma + 1));
                    }
                    catch (...)
                    {
                        std::cerr << "WARN: Failed to parse kalman-test timings, using defaults 3s/3s." << std::endl;
                    }
                }
            }
        }

        // initialize detector
        std::cout << "Loading model..." << std::endl;
        Detector detector(L"last.onnx");
        std::cout << "Model loaded successfully!" << std::endl;

        // --- VIDEO INITIALIZATION ---
        cv::VideoCapture cap("videos/test.mp4");

        if (!cap.isOpened())
        {
            std::cerr << "ERROR: Could not open video source." << std::endl;
            return -1;
        }

        // Get video properties for output
        const int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double fps_input = cap.get(cv::CAP_PROP_FPS);
        const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        // Calculate output dimensions - maintain aspect ratio, fit to OUTPUT_WIDTH x OUTPUT_HEIGHT
        const float scale = std::min(static_cast<float>(OUTPUT_WIDTH) / frame_width,
                                     static_cast<float>(OUTPUT_HEIGHT) / frame_height);
        const int scaled_width = static_cast<int>(frame_width * scale);
        const int scaled_height = static_cast<int>(frame_height * scale);
        const int offset_x = (OUTPUT_WIDTH - scaled_width) / 2;
        const int offset_y = (OUTPUT_HEIGHT - scaled_height) / 2;

        // Initialize video writer
        cv::VideoWriter writer("videos/output.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                               fps_input, cv::Size(OUTPUT_WIDTH, OUTPUT_HEIGHT));

        if (!writer.isOpened())
        {
            std::cerr << "ERROR: Could not open video writer." << std::endl;
            return -1;
        }

        std::cout << "Input video: " << frame_width << "x" << frame_height
                  << " @ " << fps_input << " FPS, " << total_frames << " frames" << std::endl;
        std::cout << "Output video: " << OUTPUT_WIDTH << "x" << OUTPUT_HEIGHT << std::endl;
        std::cout << "Writing to videos/output.mp4..." << std::endl;

        // Initialize tracking state
        std::vector<Track> activeTracks;
        int nextTrackID = 1;
        const std::vector<std::string> classNames = getClassNames();

        cv::Mat frame, resized_frame, output_frame;
        cv::Mat prev_gray, gray;
        int frame_count = 0;

        std::cout << "Processing video (no preview window)..." << std::endl;
        if (kalmanTest)
        {
            std::cout << "Kalman-only test enabled: disabling detections from "
                      << kalmanHoldStart << "s to " << (kalmanHoldStart + kalmanHoldDuration) << "s" << std::endl;
        }

        // --- THE VIDEO LOOP ---
        bool suppressDetections = false;
        bool suppressDetectionsPrev = false;
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
                OUTPUT_HEIGHT, OUTPUT_WIDTH
                {
                    for (auto &track : activeTracks)
                    {
                        track.updateWithFlow(prev_gray, gray, scaled_width, scaled_height);
                    }
                }

            // === STEP 1: Run YOLO detection (optionally suppressed for Kalman-only test window) ===
            double elapsedSeconds = (fps_input > 1e-6) ? (frame_count / fps_input) : 0.0;
            suppressDetections = kalmanTest && elapsedSeconds >= kalmanHoldStart && elapsedSeconds < (kalmanHoldStart + kalmanHoldDuration);
            if (suppressDetections != suppressDetectionsPrev)
            {
                if (suppressDetections)
                    std::cout << "\n[Kalman test] Detections OFF (" << elapsedSeconds << "s)" << std::endl;
                else
                    std::cout << "\n[Kalman test] Detections ON (" << elapsedSeconds << "s)" << std::endl;
                suppressDetectionsPrev = suppressDetections;
            }

            std::vector<Detection> results;
            if (!suppressDetections)
            {
                results = detector.run(resized_frame);
            }
            std::vector<Detection> highDetections;
            std::vector<Detection> lowDetections;
            for (const auto &det : results)
            {
                if (det.confidence >= HIGH_CONF)
                {
                    highDetections.push_back(det);
                }
                else if (det.confidence >= LOW_CONF)
                {
                    lowDetections.push_back(det);
                }
            }

            auto buildHists = [&](const std::vector<Detection> &dets)
            {
                std::vector<cv::Mat> hists;
                hists.reserve(dets.size());
                for (const auto &d : dets)
                    hists.push_back(computeHSVHist(resized_frame, d.box));
                return hists;
            };

            std::vector<cv::Mat> highHists = buildHists(highDetections);
            std::vector<cv::Mat> lowHists = buildHists(lowDetections);

            // === STEP 3: Match detections to existing tracks via Hungarian ===
            std::vector<int> trackMatchedDet(activeTracks.size(), -1);
            std::vector<bool> trackMatchIsLow(activeTracks.size(), false);
            std::vector<bool> highMatched(highDetections.size(), false);
            std::vector<bool> lowMatched(lowDetections.size(), false);

            std::vector<cv::Rect> predictedBoxes;
            predictedBoxes.reserve(activeTracks.size());
            for (auto &track : activeTracks)
            {
                cv::Rect pred = track.motion.isInitialized() ? track.motion.predict() : track.predictedBox(PREDICTION_DAMPING);
                track.lastPredBox = clampToFrame(pred, scaled_width, scaled_height);
                predictedBoxes.push_back(track.lastPredBox);
            }

            auto buildCostMatrix = [&](const std::vector<Detection> &dets, const std::vector<cv::Mat> &hists,
                                       const std::vector<int> &trackIndices)
            {
                std::vector<std::vector<float>> cost(trackIndices.size(), std::vector<float>(dets.size(), 1e6f));
                for (size_t ti = 0; ti < trackIndices.size(); ++ti)
                {
                    int t = trackIndices[ti];
                    float velMag = std::sqrt(activeTracks[t].velocity.x * activeTracks[t].velocity.x +
                                             activeTracks[t].velocity.y * activeTracks[t].velocity.y);
                    float dynamicGate = DIST_GATE_BASE + std::min(200.0f, velMag * 5.0f);
                    bool isLost = activeTracks[t].state == TrackState::Lost;
                    if (isLost)
                        dynamicGate *= 1.3f;

                    for (size_t d = 0; d < dets.size(); ++d)
                    {
                        float distance = calculateCenterDistance(predictedBoxes[t], dets[d].box);
                        float iou = calculateIOU(predictedBoxes[t], dets[d].box);

                        if (!(distance < dynamicGate || iou > IOU_GATE))
                            continue;

                        double appearanceSim = 0.0;
                        if (!activeTracks[t].appearanceHist.empty() && !hists[d].empty())
                            appearanceSim = cv::compareHist(activeTracks[t].appearanceHist, hists[d], cv::HISTCMP_CORREL);

                        // Gate very dissimilar appearance when geometry is weak
                        if (!isLost && appearanceSim < APPEARANCE_GATE && iou < 0.01f)
                            continue;

                        float classBonus = (activeTracks[t].bestClassID == dets[d].classID) ? 0.1f : 0.0f;
                        float confidenceBoost = dets[d].confidence * 0.05f;

                        // Velocity direction penalty
                        cv::Point2f predCenter(predictedBoxes[t].x + predictedBoxes[t].width / 2.0f,
                                               predictedBoxes[t].y + predictedBoxes[t].height / 2.0f);
                        cv::Point2f detCenter(dets[d].box.x + dets[d].box.width / 2.0f,
                                              dets[d].box.y + dets[d].box.height / 2.0f);
                        cv::Point2f delta = detCenter - predCenter;
                        float deltaNorm = std::sqrt(delta.x * delta.x + delta.y * delta.y);
                        float velNorm = std::sqrt(activeTracks[t].velocity.x * activeTracks[t].velocity.x + activeTracks[t].velocity.y * activeTracks[t].velocity.y);
                        float cosPenalty = 0.0f;
                        if (deltaNorm > 1e-3f && velNorm > 1e-3f)
                        {
                            float cosSim = (delta.x * activeTracks[t].velocity.x + delta.y * activeTracks[t].velocity.y) / (deltaNorm * velNorm + 1e-6f);
                            if (cosSim < 0.0f)
                                cosPenalty = COSINE_PENALTY * (-cosSim);
                        }

                        float normDist = distance / (dynamicGate + 1e-3f);
                        float baseCost = (1.0f - iou) + 0.35f * normDist + cosPenalty - static_cast<float>(0.25f * appearanceSim) - classBonus - confidenceBoost;
                        if (isLost)
                            baseCost -= 0.1f; // slightly prefer re-association for lost tracks
                        cost[ti][d] = std::max(0.0f, baseCost);
                    }
                }
                return cost;
            };

            // Primary association with high-confidence dets
            std::vector<int> allTrackIdx(activeTracks.size());
            std::iota(allTrackIdx.begin(), allTrackIdx.end(), 0);
            if (!activeTracks.empty() && !highDetections.empty())
            {
                auto highCost = buildCostMatrix(highDetections, highHists, allTrackIdx);
                auto primaryAssign = hungarianAssign(highCost, MAX_ASSIGN_COST);
                for (const auto &match : primaryAssign)
                {
                    int t = match.first;
                    int d = match.second;
                    trackMatchedDet[t] = d;
                    trackMatchIsLow[t] = false;
                    highMatched[d] = true;
                }
            }

            // Recovery association with low-confidence dets for still-unmatched tracks
            std::vector<int> unmatchedTracks;
            for (size_t t = 0; t < activeTracks.size(); ++t)
            {
                if (trackMatchedDet[t] == -1)
                    unmatchedTracks.push_back(static_cast<int>(t));
            }

            if (!unmatchedTracks.empty() && !lowDetections.empty())
            {
                auto lowCost = buildCostMatrix(lowDetections, lowHists, unmatchedTracks);
                auto recoveryAssign = hungarianAssign(lowCost, MAX_ASSIGN_COST * 0.9f);
                for (const auto &match : recoveryAssign)
                {
                    int t = unmatchedTracks[match.first];
                    int d = match.second;
                    if (trackMatchedDet[t] == -1)
                    {
                        trackMatchedDet[t] = d;
                        trackMatchIsLow[t] = true;
                        lowMatched[d] = true;
                    }
                }
            }

            // === STEP 4: Update matched tracks ===
            for (size_t t = 0; t < activeTracks.size(); t++)
            {
                if (trackMatchedDet[t] >= 0)
                {
                    bool useLow = trackMatchIsLow[t];
                    const Detection &det = useLow ? lowDetections[trackMatchedDet[t]] : highDetections[trackMatchedDet[t]];
                    const auto &histVec = useLow ? lowHists : highHists;

                    std::string aircraftName = (det.classID >= 0 && det.classID < classNames.size())
                                                   ? classNames[det.classID]
                                                   : "Unknown";
                    activeTracks[t].update(det.box, frame_count, det.confidence, det.classID, aircraftName);
                    activeTracks[t].updateAppearance(resized_frame, det.box);
                    if (trackMatchIsLow[t] && !histVec.empty())
                        activeTracks[t].appearanceHist = histVec[trackMatchedDet[t]];
                    activeTracks[t].refreshKeypoints();
                    activeTracks[t].pushTrail();
                }
            }

            // === STEP 4b: Propagate unmatched tracks forward ===
            for (size_t t = 0; t < activeTracks.size(); t++)
            {
                if (trackMatchedDet[t] == -1)
                {
                    // Use last predicted box (already from Kalman) to advance
                    activeTracks[t].lastBox = clampToFrame(activeTracks[t].lastPredBox, scaled_width, scaled_height);
                    activeTracks[t].velocity *= PREDICTION_DAMPING;
                    activeTracks[t].markStaleIfOld(frame_count, CLASS_STALE_FRAMES);
                    activeTracks[t].state = TrackState::Lost;
                    activeTracks[t].lostFrames++;
                    activeTracks[t].pushTrail();
                }
            }

            // === STEP 5: Create new tracks for unmatched high-confidence detections ===
            for (size_t d = 0; d < highDetections.size(); d++)
            {
                if (!highMatched[d])
                {
                    const Detection &det = highDetections[d];
                    std::string aircraftName = (det.classID >= 0 && det.classID < classNames.size())
                                                   ? classNames[det.classID]
                                                   : "Unknown";

                    activeTracks.emplace_back(nextTrackID++, det.box, frame_count,
                                              det.classID, aircraftName, det.confidence);
                    if (highHists.size() > d)
                        activeTracks.back().appearanceHist = highHists[d];
                    activeTracks.back().refreshKeypoints();
                    activeTracks.back().motion.initialize(det.box);
                    activeTracks.back().pushTrail();

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
                               [MAX_FRAMES_LOST](const Track &t)
                               {
                                   bool shouldRemove = t.lostFrames > MAX_FRAMES_LOST || t.state == TrackState::Removed;
                                   return shouldRemove;
                               }),
                activeTracks.end());

            // === STEP 7: Draw all active tracks ===
            int trackedCount = 0;
            int lostCount = 0;
            for (const auto &track : activeTracks)
            {
                cv::Rect adjusted_box(track.lastBox.x + offset_x, track.lastBox.y + offset_y,
                                      track.lastBox.width, track.lastBox.height);

                if (track.state == TrackState::Tracked)
                    trackedCount++;
                else if (track.state == TrackState::Lost)
                    lostCount++;

                // Draw trail (breadcrumb)
                if (track.trail.size() >= 2)
                {
                    std::vector<cv::Point> pts;
                    pts.reserve(track.trail.size());
                    for (const auto &p : track.trail)
                        pts.emplace_back(static_cast<int>(p.x) + offset_x, static_cast<int>(p.y) + offset_y);
                    cv::polylines(output_frame, pts, false, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
                }

                // Draw box
                cv::Scalar boxColor = (track.state == TrackState::Tracked) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::rectangle(output_frame, adjusted_box, boxColor, (track.state == TrackState::Tracked) ? 2 : 2, cv::LINE_AA);

                // If lost, also draw the Kalman-predicted box (ahead) to show where we think it goes
                if (track.state == TrackState::Lost && track.motion.isInitialized())
                {
                    cv::Rect predDraw = clampToFrame(track.lastPredBox, scaled_width, scaled_height);
                    predDraw.x += offset_x;
                    predDraw.y += offset_y;
                    cv::rectangle(output_frame, predDraw, cv::Scalar(0, 0, 200), 1, cv::LINE_4);
                }

                // Predicted next point marker (use last prediction when available)
                cv::Point predictedCenter;
                if (track.motion.isInitialized())
                {
                    cv::Rect p = track.lastPredBox;
                    predictedCenter = {p.x + p.width / 2 + offset_x, p.y + p.height / 2 + offset_y};
                }
                else
                {
                    predictedCenter = {adjusted_box.x + adjusted_box.width / 2, adjusted_box.y + adjusted_box.height / 2};
                }
                cv::circle(output_frame, predictedCenter, 3, cv::Scalar(0, 165, 255), cv::FILLED);

                // Draw label based on majority class and averaged confidence
                const auto [majCls, majConf] = track.majorityClass();
                const int labelCls = (majCls >= 0 && majCls < (int)classNames.size()) ? majCls : track.classID;
                const std::string &labelName = (labelCls >= 0 && labelCls < (int)classNames.size()) ? classNames[labelCls] : track.className;
                const float displayConf = std::max({track.lastConfidence, track.bestConfidence, majConf});
                const std::string label = "ID" + std::to_string(track.id) + ": " + labelName +
                                          " " + std::to_string(static_cast<int>(displayConf * 100)) + "%";

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
            const std::string trackInfo = "Tracked: " + std::to_string(trackedCount) +
                                          " | Lost: " + std::to_string(lostCount) +
                                          (kalmanTest && suppressDetections ? " | Kalman-only" : "");
            cv::putText(output_frame, trackInfo, {10, output_height - 20},
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);

            const std::string detLabel = suppressDetections ? "Detections: OFF" : "Detections: ON";
            const cv::Scalar detColor = suppressDetectOUTPUT_HEIGHT - 20
        },
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);

        const std::string detLabel = suppressDetections ? "Detections: OFF" : "Detections: ON";
        const cv::Scalar detColor = suppressDetections ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 0);
            cv::putText(output_frame, detLabel, {
            10, OUTPUT_HEIGHT

                    // carry gray frame forward for optical flow
                    prev_gray = gray.clone();
        }

        std::cout << "\nProcessing complete! Output saved to videos/output.mp4" << std::endl;
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
