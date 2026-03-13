/**
 * @file tracker.h
 * @brief 多目标跟踪器头文件
 * @author 张仕卓
 * @date 2026
 */

#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include <memory>
#include "detector.h"

/**
 * @brief 跟踪目标结构体
 */
struct TrackedObject {
    int id;                 // 目标ID
    DetectionBox box;       // 当前边界框
    int class_id;           // 类别ID
    float velocity_x;       // x方向速度
    float velocity_y;       // y方向速度
    int age;                // 轨迹年龄（帧数）
    int hits;               // 连续命中次数
    int time_since_update;  // 自上次更新的帧数
};

/**
 * @brief 多目标跟踪器类
 * 
 * 实现基于卡尔曼滤波的多目标跟踪
 */
class Tracker {
public:
    /**
     * @brief 构造函数
     * @param max_age 轨迹最大消失帧数
     * @param min_hits 确认轨迹所需最小命中数
     * @param iou_threshold IoU匹配阈值
     */
    Tracker(int max_age = 30, int min_hits = 3, float iou_threshold = 0.3f);
    
    /**
     * @brief 析构函数
     */
    ~Tracker();
    
    /**
     * @brief 更新跟踪器
     * @param detections 当前帧检测结果
     * @return 当前跟踪的目标列表
     */
    std::vector<TrackedObject> update(const std::vector<DetectionBox>& detections);
    
    /**
     * @brief 获取当前活跃轨迹数
     * @return 活跃轨迹数量
     */
    int getActiveTrackCount() const;
    
    /**
     * @brief 重置跟踪器
     */
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
    int max_age_;           // 最大消失帧数
    int min_hits_;          // 最小命中次数
    float iou_threshold_;   // IoU阈值
    int next_id_;           // 下一个分配的ID
};

#endif // TRACKER_H
