/**
 * @file pipeline.h
 * @brief 处理流水线头文件
 * @author 张仕卓
 * @date 2026
 */

#ifndef PIPELINE_H
#define PIPELINE_H

#include <string>
#include <memory>
#include <functional>
#include "detector.h"
#include "tracker.h"

/**
 * @brief 帧处理结果结构体
 */
struct FrameResult {
    int frame_id;                           // 帧ID
    std::vector<TrackedObject> objects;     // 跟踪目标列表
    float preprocess_time;                  // 预处理时间(ms)
    float inference_time;                   // 推理时间(ms)
    float postprocess_time;                 // 后处理时间(ms)
    float tracking_time;                    // 跟踪时间(ms)
    float total_time;                       // 总时间(ms)
};

/**
 * @brief 结果回调函数类型
 */
using ResultCallback = std::function<void(const FrameResult&)>;

/**
 * @brief 处理流水线类
 * 
 * 集成检测、跟踪和可视化的完整处理流水线
 */
class Pipeline {
public:
    /**
     * @brief 构造函数
     * @param config_path 配置文件路径
     */
    explicit Pipeline(const std::string& config_path);
    
    /**
     * @brief 析构函数
     */
    ~Pipeline();
    
    /**
     * @brief 初始化流水线
     * @return 是否成功
     */
    bool init();
    
    /**
     * @brief 处理单帧图像
     * @param image 图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @return 处理结果
     */
    FrameResult processFrame(const unsigned char* image, int width, int height);
    
    /**
     * @brief 开始处理视频流
     * @param source 视频源（设备路径或文件路径）
     */
    void startVideoStream(const std::string& source);
    
    /**
     * @brief 停止视频流处理
     */
    void stopVideoStream();
    
    /**
     * @brief 设置结果回调
     * @param callback 回调函数
     */
    void setResultCallback(ResultCallback callback);
    
    /**
     * @brief 获取当前FPS
     * @return 帧率
     */
    float getCurrentFPS() const;
    
    /**
     * @brief 获取统计信息
     * @return 统计信息字符串
     */
    std::string getStats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // PIPELINE_H
