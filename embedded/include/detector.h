/**
 * @file detector.h
 * @brief 红外目标检测器头文件
 * @author 张仕卓
 * @date 2026
 */

#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>
#include <vector>
#include <memory>

/**
 * @brief 检测框结构体
 */
struct DetectionBox {
    float x;        // 左上角x坐标
    float y;        // 左上角y坐标
    float width;    // 宽度
    float height;   // 高度
    float score;    // 置信度
    int class_id;   // 类别ID
};

/**
 * @brief 红外目标检测器类
 * 
 * 使用RKNN进行目标检测推理
 */
class Detector {
public:
    /**
     * @brief 构造函数
     * @param model_path RKNN模型路径
     */
    explicit Detector(const std::string& model_path);
    
    /**
     * @brief 析构函数
     */
    ~Detector();
    
    /**
     * @brief 初始化检测器
     * @return 是否成功
     */
    bool init();
    
    /**
     * @brief 执行目标检测
     * @param image 输入图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param detections 输出检测结果
     * @return 是否成功
     */
    bool detect(const unsigned char* image, int width, int height,
                std::vector<DetectionBox>& detections);
    
    /**
     * @brief 设置置信度阈值
     * @param threshold 置信度阈值
     */
    void setConfThreshold(float threshold);
    
    /**
     * @brief 设置NMS阈值
     * @param threshold NMS IoU阈值
     */
    void setNMSThreshold(float threshold);
    
    /**
     * @brief 获取推理时间（毫秒）
     * @return 上一次推理耗时
     */
    float getInferenceTime() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;  // PIMPL模式
    
    float conf_threshold_;  // 置信度阈值
    float nms_threshold_;   // NMS阈值
    float inference_time_;  // 推理时间
};

#endif // DETECTOR_H
