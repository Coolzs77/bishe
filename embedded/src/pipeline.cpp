/**
 * @file pipeline.cpp
 * @brief 检测跟踪处理流水线实现
 * 
 * 整合检测器和跟踪器的完整处理流程
 */

#include "pipeline.h"
#include <iostream>
#include <iomanip>

namespace infrared {

// ==================== DetectionTrackingPipeline 实现 ====================

DetectionTrackingPipeline::DetectionTrackingPipeline()
    : is_initialized_(false)
    , frame_count_(0)
    , total_detect_time_(0)
    , total_track_time_(0)
    , total_process_time_(0) {
    
    detector_ = std::make_unique<YOLOv5Detector>();
    tracker_ = std::make_unique<ByteTracker>();
}

DetectionTrackingPipeline::~DetectionTrackingPipeline() {
    release();
}

bool DetectionTrackingPipeline::initialize(const PipelineConfig& config) {
    config_ = config;
    
    std::cout << "Initializing detection and tracking pipeline..." << std::endl;
    
    // 初始化检测器
    std::cout << "  Loading detection model: " << config.model_path << std::endl;
    if (!detector_->initialize(config.model_path, config.input_width, config.input_height)) {
        std::cerr << "Detector initialization failed" << std::endl;
        return false;
    }
    
    detector_->setConfidenceThreshold(config.conf_threshold);
    detector_->setNMSThreshold(config.nms_threshold);
    
    if (!config.class_names.empty()) {
        detector_->setClassNames(config.class_names);
    }
    
    // 初始化跟踪器
    std::cout << "  Initializing tracker..." << std::endl;
    tracker_->initialize(
        config.track_thresh,
        config.low_thresh,
        config.match_thresh,
        config.max_time_lost
    );
    
    is_initialized_ = true;
    frame_count_ = 0;
    total_detect_time_ = 0;
    total_track_time_ = 0;
    total_process_time_ = 0;
    
    std::cout << "Pipeline initialization completed" << std::endl;
    
    return true;
}

void DetectionTrackingPipeline::release() {
    if (detector_) {
        detector_->release();
    }
    if (tracker_) {
        tracker_->reset();
    }
    is_initialized_ = false;
}

std::vector<TrackResult> DetectionTrackingPipeline::process(const cv::Mat& frame) {
    if (!is_initialized_) {
        std::cerr << "Pipeline not initialized" << std::endl;
        return {};
    }
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 检测
    auto detect_start = std::chrono::high_resolution_clock::now();
    std::vector<DetectionResult> detections = detector_->detect(frame);
    auto detect_end = std::chrono::high_resolution_clock::now();
    
    // 跟踪
    auto track_start = std::chrono::high_resolution_clock::now();
    std::vector<TrackResult> tracks = tracker_->update(detections);
    auto track_end = std::chrono::high_resolution_clock::now();
    
    auto total_end = std::chrono::high_resolution_clock::now();
    
    // 统计时间
    double detect_time = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
    double track_time = std::chrono::duration<double, std::milli>(track_end - track_start).count();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    total_detect_time_ += detect_time;
    total_track_time_ += track_time;
    total_process_time_ += total_time;
    frame_count_++;
    
    return tracks;
}

cv::Mat DetectionTrackingPipeline::visualize(
    const cv::Mat& frame,
    const std::vector<TrackResult>& tracks,
    bool draw_trajectory,
    int trajectory_length) {
    
    cv::Mat vis = frame.clone();
    
    // 颜色表
    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),    // 蓝
        cv::Scalar(0, 255, 0),    // 绿
        cv::Scalar(0, 0, 255),    // 红
        cv::Scalar(255, 255, 0),  // 青
        cv::Scalar(255, 0, 255),  // 品红
        cv::Scalar(0, 255, 255),  // 黄
        cv::Scalar(128, 0, 255),  // 紫
        cv::Scalar(255, 128, 0),  // 橙
    };
    
    for (const auto& track : tracks) {
        // 获取颜色
        cv::Scalar color = colors[track.track_id % colors.size()];
        
        // 绘制边界框
        cv::rectangle(vis,
                     cv::Point(track.x1, track.y1),
                     cv::Point(track.x2, track.y2),
                     color, 2);
        
        // 绘制ID标签
        std::string label = "ID:" + std::to_string(track.track_id);
        
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        
        cv::rectangle(vis,
                     cv::Point(track.x1, track.y1 - text_size.height - 10),
                     cv::Point(track.x1 + text_size.width + 10, track.y1),
                     color, -1);
        
        cv::putText(vis, label,
                   cv::Point(track.x1 + 5, track.y1 - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6,
                   cv::Scalar(255, 255, 255), 1);
        
        // 绘制轨迹
        if (draw_trajectory && track.trajectory.size() > 1) {
            int start_idx = std::max(0, static_cast<int>(track.trajectory.size()) - trajectory_length);
            
            for (size_t i = start_idx + 1; i < track.trajectory.size(); ++i) {
                float alpha = static_cast<float>(i - start_idx) / trajectory_length;
                cv::Scalar traj_color = color * alpha;
                
                cv::line(vis,
                        cv::Point(track.trajectory[i - 1].first, track.trajectory[i - 1].second),
                        cv::Point(track.trajectory[i].first, track.trajectory[i].second),
                        traj_color, 2);
            }
        }
    }
    
    return vis;
}

void DetectionTrackingPipeline::reset() {
    if (tracker_) {
        tracker_->reset();
    }
    frame_count_ = 0;
    total_detect_time_ = 0;
    total_track_time_ = 0;
    total_process_time_ = 0;
}

PipelineStats DetectionTrackingPipeline::getStats() const {
    PipelineStats stats;
    stats.frame_count = frame_count_;
    
    if (frame_count_ > 0) {
        stats.avg_detect_time_ms = total_detect_time_ / frame_count_;
        stats.avg_track_time_ms = total_track_time_ / frame_count_;
        stats.avg_total_time_ms = total_process_time_ / frame_count_;
        stats.fps = 1000.0 / stats.avg_total_time_ms;
    } else {
        stats.avg_detect_time_ms = 0;
        stats.avg_track_time_ms = 0;
        stats.avg_total_time_ms = 0;
        stats.fps = 0;
    }
    
    return stats;
}

void DetectionTrackingPipeline::printStats() const {
    PipelineStats stats = getStats();
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "========================" << std::endl;
    std::cout << "Pipeline Performance Stats" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Frames processed: " << stats.frame_count << std::endl;
    std::cout << "Avg detection time: " << stats.avg_detect_time_ms << " ms" << std::endl;
    std::cout << "Avg tracking time: " << stats.avg_track_time_ms << " ms" << std::endl;
    std::cout << "Avg total time: " << stats.avg_total_time_ms << " ms" << std::endl;
    std::cout << "Avg FPS: " << stats.fps << std::endl;
    std::cout << "========================" << std::endl;
}

// ==================== 便捷函数实现 ====================

std::unique_ptr<DetectionTrackingPipeline> createPipeline(const std::string& config_path) {
    auto pipeline = std::make_unique<DetectionTrackingPipeline>();
    
    // 从配置文件加载配置
    PipelineConfig config;
    
    // 尝试读取YAML配置
    try {
        cv::FileStorage fs(config_path, cv::FileStorage::READ);
        if (fs.isOpened()) {
            fs["model_path"] >> config.model_path;
            fs["input_width"] >> config.input_width;
            fs["input_height"] >> config.input_height;
            fs["conf_threshold"] >> config.conf_threshold;
            fs["nms_threshold"] >> config.nms_threshold;
            fs["track_thresh"] >> config.track_thresh;
            fs["low_thresh"] >> config.low_thresh;
            fs["match_thresh"] >> config.match_thresh;
            fs["max_time_lost"] >> config.max_time_lost;
            
            cv::FileNode classes = fs["class_names"];
            if (classes.type() == cv::FileNode::SEQ) {
                for (auto it = classes.begin(); it != classes.end(); ++it) {
                    config.class_names.push_back(static_cast<std::string>(*it));
                }
            }
            
            fs.release();
        } else {
            std::cerr << "Cannot open config file: " << config_path << std::endl;
            std::cout << "Using default configuration" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading config file: " << e.what() << std::endl;
        std::cout << "Using default configuration" << std::endl;
    }
    
    // 使用默认值填充未设置的参数
    if (config.model_path.empty()) {
        config.model_path = "models/yolov5s_infrared.rknn";
    }
    if (config.input_width == 0) config.input_width = 640;
    if (config.input_height == 0) config.input_height = 640;
    if (config.conf_threshold == 0) config.conf_threshold = 0.5f;
    if (config.nms_threshold == 0) config.nms_threshold = 0.45f;
    if (config.track_thresh == 0) config.track_thresh = 0.5f;
    if (config.low_thresh == 0) config.low_thresh = 0.1f;
    if (config.match_thresh == 0) config.match_thresh = 0.8f;
    if (config.max_time_lost == 0) config.max_time_lost = 30;
    if (config.class_names.empty()) {
        config.class_names = {"person", "car", "bicycle"};
    }
    
    if (!pipeline->initialize(config)) {
        return nullptr;
    }
    
    return pipeline;
}

std::unique_ptr<DetectionTrackingPipeline> createPipelineWithConfig(const PipelineConfig& config) {
    auto pipeline = std::make_unique<DetectionTrackingPipeline>();
    
    if (!pipeline->initialize(config)) {
        return nullptr;
    }
    
    return pipeline;
}

} // namespace infrared
