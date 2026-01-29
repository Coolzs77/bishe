/**
 * @file detector.cpp
 * @brief RKNN YOLOv5 目标检测器实现
 * 
 * 用于RV1126平台的红外目标检测
 */

#include "detector.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>

namespace infrared {

// ==================== DetectionResult 实现 ====================

std::pair<float, float> DetectionResult::getCenter() const {
    return {(x1 + x2) / 2.0f, (y1 + y2) / 2.0f};
}

std::pair<float, float> DetectionResult::getSize() const {
    return {x2 - x1, y2 - y1};
}

float DetectionResult::getArea() const {
    return (x2 - x1) * (y2 - y1);
}

// ==================== YOLOv5Detector 实现 ====================

YOLOv5Detector::YOLOv5Detector()
    : rknn_ctx_(0)
    , input_width_(640)
    , input_height_(640)
    , conf_threshold_(0.5f)
    , nms_threshold_(0.45f)
    , is_initialized_(false) {
    
    // 默认类别名称
    class_names_ = {"person", "car", "bicycle"};
    num_classes_ = class_names_.size();
}

YOLOv5Detector::~YOLOv5Detector() {
    release();
}

bool YOLOv5Detector::initialize(const std::string& model_path, int input_width, int input_height) {
    if (is_initialized_) {
        release();
    }
    
    input_width_ = input_width;
    input_height_ = input_height;
    
#ifdef USE_RKNN
    // 读取RKNN模型文件
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return false;
    }
    
    size_t model_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> model_data(model_size);
    file.read(model_data.data(), model_size);
    file.close();
    
    // 初始化RKNN
    int ret = rknn_init(&rknn_ctx_, model_data.data(), model_size, 0, nullptr);
    if (ret < 0) {
        std::cerr << "RKNN initialization failed: " << ret << std::endl;
        return false;
    }
    
    // 获取输入输出属性
    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        std::cerr << "Failed to query input/output count" << std::endl;
        rknn_destroy(rknn_ctx_);
        return false;
    }
    
    std::cout << "Model loaded successfully: " << model_path << std::endl;
    std::cout << "  Input count: " << io_num.n_input << std::endl;
    std::cout << "  Output count: " << io_num.n_output << std::endl;
    
    is_initialized_ = true;
    return true;
    
#else
    std::cout << "RKNN not enabled, using simulation mode" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    is_initialized_ = true;
    return true;
#endif
}

void YOLOv5Detector::release() {
#ifdef USE_RKNN
    if (rknn_ctx_ != 0) {
        rknn_destroy(rknn_ctx_);
        rknn_ctx_ = 0;
    }
#endif
    is_initialized_ = false;
}

void YOLOv5Detector::setConfidenceThreshold(float threshold) {
    conf_threshold_ = threshold;
}

void YOLOv5Detector::setNMSThreshold(float threshold) {
    nms_threshold_ = threshold;
}

void YOLOv5Detector::setClassNames(const std::vector<std::string>& names) {
    class_names_ = names;
    num_classes_ = names.size();
}

std::vector<DetectionResult> YOLOv5Detector::detect(const cv::Mat& image) {
    if (!is_initialized_) {
        std::cerr << "Detector not initialized" << std::endl;
        return {};
    }
    
    if (image.empty()) {
        std::cerr << "Input image is empty" << std::endl;
        return {};
    }
    
    // 预处理
    cv::Mat preprocessed = preprocess(image);
    
    // 推理
    std::vector<float> output = inference(preprocessed);
    
    // 后处理
    return postprocess(output, image.cols, image.rows);
}

cv::Mat YOLOv5Detector::preprocess(const cv::Mat& image) {
    cv::Mat resized;
    
    // 保持宽高比的letterbox缩放
    float scale = std::min(
        static_cast<float>(input_width_) / image.cols,
        static_cast<float>(input_height_) / image.rows
    );
    
    int new_width = static_cast<int>(image.cols * scale);
    int new_height = static_cast<int>(image.rows * scale);
    
    cv::resize(image, resized, cv::Size(new_width, new_height));
    
    // 创建letterbox
    cv::Mat padded(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    
    int dx = (input_width_ - new_width) / 2;
    int dy = (input_height_ - new_height) / 2;
    
    resized.copyTo(padded(cv::Rect(dx, dy, new_width, new_height)));
    
    // 保存缩放信息用于后处理
    scale_ = scale;
    pad_x_ = dx;
    pad_y_ = dy;
    
    // BGR转RGB
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
    
    return padded;
}

std::vector<float> YOLOv5Detector::inference(const cv::Mat& input) {
#ifdef USE_RKNN
    // 设置输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = input.total() * input.elemSize();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input.data;
    
    int ret = rknn_inputs_set(rknn_ctx_, 1, inputs);
    if (ret < 0) {
        std::cerr << "Failed to set input" << std::endl;
        return {};
    }
    
    // 运行推理
    ret = rknn_run(rknn_ctx_, nullptr);
    if (ret < 0) {
        std::cerr << "Inference failed" << std::endl;
        return {};
    }
    
    // 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    
    ret = rknn_outputs_get(rknn_ctx_, 1, outputs, nullptr);
    if (ret < 0) {
        std::cerr << "Failed to get output" << std::endl;
        return {};
    }
    
    // 复制输出数据
    int output_size = outputs[0].size / sizeof(float);
    std::vector<float> result(output_size);
    memcpy(result.data(), outputs[0].buf, outputs[0].size);
    
    rknn_outputs_release(rknn_ctx_, 1, outputs);
    
    return result;
    
#else
    // 模拟推理 - 返回空结果
    return {};
#endif
}

std::vector<DetectionResult> YOLOv5Detector::postprocess(
    const std::vector<float>& output,
    int orig_width,
    int orig_height) {
    
    std::vector<DetectionResult> results;
    
    if (output.empty()) {
        return results;
    }
    
    // 假设输出格式为 [N, 5+num_classes]
    // 每行: [x, y, w, h, obj_conf, cls1_conf, cls2_conf, ...]
    int stride = 5 + num_classes_;
    int num_proposals = output.size() / stride;
    
    std::vector<DetectionResult> candidates;
    
    for (int i = 0; i < num_proposals; ++i) {
        const float* row = &output[i * stride];
        
        float obj_conf = row[4];
        if (obj_conf < conf_threshold_) {
            continue;
        }
        
        // 找最大类别概率
        int best_class = 0;
        float best_score = 0;
        for (int c = 0; c < num_classes_; ++c) {
            float score = row[5 + c] * obj_conf;
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }
        
        if (best_score < conf_threshold_) {
            continue;
        }
        
        // 转换坐标 (中心点+宽高 -> 左上右下)
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        
        DetectionResult det;
        det.x1 = cx - w / 2;
        det.y1 = cy - h / 2;
        det.x2 = cx + w / 2;
        det.y2 = cy + h / 2;
        det.confidence = best_score;
        det.class_id = best_class;
        det.class_name = (best_class < class_names_.size()) ? class_names_[best_class] : "unknown";
        
        // 映射回原始图像坐标
        det.x1 = (det.x1 - pad_x_) / scale_;
        det.y1 = (det.y1 - pad_y_) / scale_;
        det.x2 = (det.x2 - pad_x_) / scale_;
        det.y2 = (det.y2 - pad_y_) / scale_;
        
        // 裁剪到有效范围
        det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(orig_width)));
        det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(orig_height)));
        det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(orig_width)));
        det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(orig_height)));
        
        candidates.push_back(det);
    }
    
    // NMS
    results = applyNMS(candidates);
    
    return results;
}

std::vector<DetectionResult> YOLOv5Detector::applyNMS(
    const std::vector<DetectionResult>& detections) {
    
    if (detections.empty()) {
        return {};
    }
    
    // 按置信度排序
    std::vector<int> indices(detections.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    
    std::sort(indices.begin(), indices.end(), [&detections](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });
    
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<DetectionResult> results;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) {
            continue;
        }
        
        results.push_back(detections[idx]);
        
        // 抑制重叠框
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx2 = indices[j];
            if (suppressed[idx2]) {
                continue;
            }
            
            // 只对同类别进行NMS
            if (detections[idx].class_id != detections[idx2].class_id) {
                continue;
            }
            
            float iou = computeIoU(detections[idx], detections[idx2]);
            if (iou > nms_threshold_) {
                suppressed[idx2] = true;
            }
        }
    }
    
    return results;
}

float YOLOv5Detector::computeIoU(const DetectionResult& a, const DetectionResult& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    
    float inter_width = std::max(0.0f, x2 - x1);
    float inter_height = std::max(0.0f, y2 - y1);
    float inter_area = inter_width * inter_height;
    
    float area_a = a.getArea();
    float area_b = b.getArea();
    float union_area = area_a + area_b - inter_area;
    
    if (union_area <= 0) {
        return 0.0f;
    }
    
    return inter_area / union_area;
}

} // namespace infrared
