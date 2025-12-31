#include "detector.h"
#include <iostream>

namespace
{
    constexpr int INPUT_SIZE = 640;
    constexpr int NUM_BOXES = 8400;
    constexpr int NUM_CLASSES = 60;
    constexpr float DETECTION_THRESHOLD = 0.25f;
    constexpr float NMS_CONF_THRESHOLD = 0.2f;
    constexpr float NMS_IOU_THRESHOLD = 0.45f;
    constexpr int CHANNELS = 3;
    constexpr float NORM_FACTOR = 1.0f / 255.0f;
}

Detector::Detector(const std::wstring &modelPath)
{
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "AeroLog");
    Ort::SessionOptions options;

    // Try to use CUDA (GPU) if available, fall back to CPU
    bool cudaAvailable = false;
    try
    {
        std::vector<std::string> providers = Ort::GetAvailableProviders();
        auto it = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider");
        if (it != providers.end())
        {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024; // 2GB
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;

            options.AppendExecutionProvider_CUDA(cuda_options);
            cudaAvailable = true;
            std::cout << "Using CUDA GPU acceleration" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "CUDA initialization failed: " << e.what() << std::endl;
    }

    if (!cudaAvailable)
    {
        std::cout << "Using CPU inference" << std::endl;
    }

    // Multi-threading for CPU fallback
    options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    session = Ort::Session(env, modelPath.c_str(), options);
}

std::vector<Detection> Detector::run(cv::Mat &frame)
{
    // Pre-process: resize and pad to INPUT_SIZE x INPUT_SIZE
    const float scale = std::min(static_cast<float>(INPUT_SIZE) / frame.cols,
                                 static_cast<float>(INPUT_SIZE) / frame.rows);
    const float inv_scale = 1.0f / scale;
    const int nw = static_cast<int>(frame.cols * scale);
    const int nh = static_cast<int>(frame.rows * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh));

    // Create canvas and convert BGR to RGB
    cv::Mat canvas(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(0, 0, nw, nh)));
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    // Convert HWC to CHW format with normalization
    constexpr int TENSOR_SIZE = CHANNELS * INPUT_SIZE * INPUT_SIZE;
    std::vector<float> inputTensorValues(TENSOR_SIZE);

    // Use pointer access for 3-5x speedup over .at<>()
    constexpr int AREA = INPUT_SIZE * INPUT_SIZE;
    for (int h = 0; h < INPUT_SIZE; ++h)
    {
        const uchar *row_ptr = canvas.ptr<uchar>(h);
        const int row_offset = h * INPUT_SIZE;
        for (int w = 0; w < INPUT_SIZE; ++w)
        {
            const int pixel_idx = w * CHANNELS;
            const int idx = row_offset + w;
            inputTensorValues[idx] = row_ptr[pixel_idx] * NORM_FACTOR;                // R
            inputTensorValues[AREA + idx] = row_ptr[pixel_idx + 1] * NORM_FACTOR;     // G
            inputTensorValues[2 * AREA + idx] = row_ptr[pixel_idx + 2] * NORM_FACTOR; // B
        }
    }

    // Run inference
    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    constexpr std::array<int64_t, 4> inputShape = {1, CHANNELS, INPUT_SIZE, INPUT_SIZE};

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memory_info, inputTensorValues.data(), inputTensorValues.size(),
        inputShape.data(), inputShape.size());

    constexpr const char *inputNames[] = {"images"};
    constexpr const char *outputNames[] = {"output0"};

    auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
    const float *rawOutput = outputTensors[0].GetTensorMutableData<float>();

    // Post-process: extract boxes, scores, and class IDs
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;
    boxes.reserve(NUM_BOXES / 10); // Reserve for ~10% detection rate
    confs.reserve(NUM_BOXES / 10);
    classIds.reserve(NUM_BOXES / 10);

    for (int i = 0; i < NUM_BOXES; ++i)
    {
        const float *classes_scores = rawOutput + 4 * NUM_BOXES + i;

        // Find class with highest score
        float max_score = 0.0f;
        int class_id = 0;
        for (int cls = 0; cls < NUM_CLASSES; ++cls)
        {
            const float score = classes_scores[cls * NUM_BOXES];
            if (score > max_score)
            {
                max_score = score;
                class_id = cls;
            }
        }

        if (max_score > DETECTION_THRESHOLD)
        {
            const float cx = rawOutput[i];                // Center X
            const float cy = rawOutput[NUM_BOXES + i];    // Center Y
            const float w = rawOutput[2 * NUM_BOXES + i]; // Width
            const float h = rawOutput[3 * NUM_BOXES + i]; // Height

            // Convert YOLO coordinates back to original frame coordinates
            const float half_w = w * 0.5f;
            const float half_h = h * 0.5f;
            const int left = static_cast<int>((cx - half_w) * inv_scale);
            const int top = static_cast<int>((cy - half_h) * inv_scale);
            const int width = static_cast<int>(w * inv_scale);
            const int height = static_cast<int>(h * inv_scale);

            boxes.emplace_back(left, top, width, height);
            confs.push_back(max_score);
            classIds.push_back(class_id);
        }
    }

    // Apply Non-Maximum Suppression to remove duplicate detections
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, NMS_CONF_THRESHOLD, NMS_IOU_THRESHOLD, indices);

    std::vector<Detection> final_results;
    final_results.reserve(indices.size());
    for (const int idx : indices)
    {
        final_results.push_back({boxes[idx], confs[idx], classIds[idx]});
    }

    return final_results;
}