/**
 * @file tracker.cpp
 * @brief 多目标跟踪器实现
 * 
 * ByteTrack算法的嵌入式实现
 */

#include "tracker.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace infrared {

// ==================== Track 实现 ====================

Track::Track(const DetectionResult& det, int id)
    : track_id(id)
    , state(TrackState::New)
    , hits(1)
    , time_since_update(0)
    , age(0)
    , class_id(det.class_id) {
    
    // 初始化边界框
    bbox[0] = det.x1;
    bbox[1] = det.y1;
    bbox[2] = det.x2;
    bbox[3] = det.y2;
    
    // 初始化卡尔曼滤波器状态
    // 状态: [cx, cy, a, h, vx, vy, va, vh]
    float cx = (det.x1 + det.x2) / 2;
    float cy = (det.y1 + det.y2) / 2;
    float w = det.x2 - det.x1;
    float h = det.y2 - det.y1;
    float a = w / h;  // 宽高比
    
    mean = Eigen::VectorXf(8);
    mean << cx, cy, a, h, 0, 0, 0, 0;
    
    // 初始化协方差
    covariance = Eigen::MatrixXf::Identity(8, 8);
    covariance(0, 0) = 2 * 0.05f * h;
    covariance(1, 1) = 2 * 0.05f * h;
    covariance(2, 2) = 1e-2;
    covariance(3, 3) = 2 * 0.05f * h;
    covariance(4, 4) = 10 * 0.00625f * h;
    covariance(5, 5) = 10 * 0.00625f * h;
    covariance(6, 6) = 1e-5;
    covariance(7, 7) = 10 * 0.00625f * h;
    
    // 记录轨迹
    trajectory.push_back({cx, cy});
}

void Track::predict() {
    // 状态转移矩阵 F
    Eigen::MatrixXf F = Eigen::MatrixXf::Identity(8, 8);
    for (int i = 0; i < 4; ++i) {
        F(i, i + 4) = 1.0f;
    }
    
    // 过程噪声
    float h = mean(3);
    Eigen::VectorXf std_pos(4);
    std_pos << 0.05f * h, 0.05f * h, 1e-2, 0.05f * h;
    
    Eigen::VectorXf std_vel(4);
    std_vel << 0.00625f * h, 0.00625f * h, 1e-5, 0.00625f * h;
    
    Eigen::VectorXf std_all(8);
    std_all << std_pos, std_vel;
    
    Eigen::MatrixXf Q = std_all.array().square().matrix().asDiagonal();
    
    // 预测
    mean = F * mean;
    covariance = F * covariance * F.transpose() + Q;
    
    age++;
    time_since_update++;
    
    // 更新边界框
    updateBBox();
}

void Track::update(const DetectionResult& det) {
    // 观测
    float cx = (det.x1 + det.x2) / 2;
    float cy = (det.y1 + det.y2) / 2;
    float w = det.x2 - det.x1;
    float h = det.y2 - det.y1;
    float a = w / h;
    
    Eigen::VectorXf measurement(4);
    measurement << cx, cy, a, h;
    
    // 观测矩阵 H
    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(4, 8);
    for (int i = 0; i < 4; ++i) {
        H(i, i) = 1.0f;
    }
    
    // 观测噪声
    Eigen::VectorXf std_obs(4);
    std_obs << 0.05f * mean(3), 0.05f * mean(3), 1e-1, 0.05f * mean(3);
    Eigen::MatrixXf R = std_obs.array().square().matrix().asDiagonal();
    
    // 卡尔曼增益
    Eigen::MatrixXf S = H * covariance * H.transpose() + R;
    Eigen::MatrixXf K = covariance * H.transpose() * S.inverse();
    
    // 更新
    Eigen::VectorXf innovation = measurement - H * mean;
    mean = mean + K * innovation;
    covariance = (Eigen::MatrixXf::Identity(8, 8) - K * H) * covariance;
    
    hits++;
    time_since_update = 0;
    
    // 更新状态
    if (state == TrackState::New && hits >= 3) {
        state = TrackState::Tracked;
    }
    
    // 更新边界框和轨迹
    updateBBox();
    trajectory.push_back({mean(0), mean(1)});
    
    // 限制轨迹长度
    if (trajectory.size() > 100) {
        trajectory.erase(trajectory.begin());
    }
}

void Track::updateBBox() {
    float cx = mean(0);
    float cy = mean(1);
    float a = mean(2);
    float h = mean(3);
    float w = a * h;
    
    bbox[0] = cx - w / 2;
    bbox[1] = cy - h / 2;
    bbox[2] = cx + w / 2;
    bbox[3] = cy + h / 2;
}

void Track::markLost() {
    state = TrackState::Lost;
}

void Track::markRemoved() {
    state = TrackState::Removed;
}

// ==================== ByteTracker 实现 ====================

ByteTracker::ByteTracker()
    : track_thresh_(0.5f)
    , low_thresh_(0.1f)
    , match_thresh_(0.8f)
    , max_time_lost_(30)
    , next_id_(1) {
}

void ByteTracker::initialize(float track_thresh, float low_thresh, 
                             float match_thresh, int max_time_lost) {
    track_thresh_ = track_thresh;
    low_thresh_ = low_thresh;
    match_thresh_ = match_thresh;
    max_time_lost_ = max_time_lost;
    reset();
}

void ByteTracker::reset() {
    tracked_tracks_.clear();
    lost_tracks_.clear();
    next_id_ = 1;
}

std::vector<TrackResult> ByteTracker::update(const std::vector<DetectionResult>& detections) {
    // 分离高低置信度检测
    std::vector<const DetectionResult*> high_conf_dets;
    std::vector<const DetectionResult*> low_conf_dets;
    
    for (const auto& det : detections) {
        if (det.confidence >= track_thresh_) {
            high_conf_dets.push_back(&det);
        } else if (det.confidence >= low_thresh_) {
            low_conf_dets.push_back(&det);
        }
    }
    
    // 预测所有轨迹
    for (auto& track : tracked_tracks_) {
        track->predict();
    }
    for (auto& track : lost_tracks_) {
        track->predict();
    }
    
    // 合并跟踪和丢失轨迹池
    std::vector<std::shared_ptr<Track>> track_pool;
    track_pool.insert(track_pool.end(), tracked_tracks_.begin(), tracked_tracks_.end());
    track_pool.insert(track_pool.end(), lost_tracks_.begin(), lost_tracks_.end());
    
    // 第一次关联：高置信度检测与所有轨迹
    std::vector<std::shared_ptr<Track>> activated_tracks;
    std::vector<std::shared_ptr<Track>> refound_tracks;
    std::vector<std::shared_ptr<Track>> unmatched_tracks;
    std::vector<int> unmatched_det_indices;
    
    if (!high_conf_dets.empty() && !track_pool.empty()) {
        // 计算IoU代价矩阵
        Eigen::MatrixXf cost_matrix = computeCostMatrix(track_pool, high_conf_dets);
        
        // 匹配
        std::vector<std::pair<int, int>> matches;
        std::vector<int> unmatched_track_indices;
        linearAssignment(cost_matrix, 1.0f - match_thresh_, 
                        matches, unmatched_track_indices, unmatched_det_indices);
        
        // 处理匹配
        for (const auto& match : matches) {
            auto& track = track_pool[match.first];
            const auto* det = high_conf_dets[match.second];
            
            track->update(*det);
            
            if (track->state == TrackState::Lost) {
                refound_tracks.push_back(track);
            } else {
                activated_tracks.push_back(track);
            }
        }
        
        // 收集未匹配轨迹
        for (int idx : unmatched_track_indices) {
            unmatched_tracks.push_back(track_pool[idx]);
        }
    } else {
        unmatched_tracks = track_pool;
        for (size_t i = 0; i < high_conf_dets.size(); ++i) {
            unmatched_det_indices.push_back(i);
        }
    }
    
    // 第二次关联：低置信度检测与剩余跟踪轨迹
    std::vector<std::shared_ptr<Track>> remaining_tracked;
    for (auto& track : unmatched_tracks) {
        if (track->state == TrackState::Tracked) {
            remaining_tracked.push_back(track);
        }
    }
    
    if (!low_conf_dets.empty() && !remaining_tracked.empty()) {
        Eigen::MatrixXf cost_matrix = computeCostMatrix(remaining_tracked, low_conf_dets);
        
        std::vector<std::pair<int, int>> matches;
        std::vector<int> unmatched_track_indices;
        std::vector<int> unmatched_low_det_indices;
        linearAssignment(cost_matrix, 0.5f, matches, 
                        unmatched_track_indices, unmatched_low_det_indices);
        
        for (const auto& match : matches) {
            auto& track = remaining_tracked[match.first];
            const auto* det = low_conf_dets[match.second];
            track->update(*det);
            activated_tracks.push_back(track);
        }
        
        // 标记未匹配轨迹为丢失
        for (int idx : unmatched_track_indices) {
            remaining_tracked[idx]->markLost();
        }
    } else {
        for (auto& track : remaining_tracked) {
            track->markLost();
        }
    }
    
    // 为未匹配的高置信度检测创建新轨迹
    for (int idx : unmatched_det_indices) {
        const auto* det = high_conf_dets[idx];
        if (det->confidence >= track_thresh_) {
            auto new_track = std::make_shared<Track>(*det, next_id_++);
            activated_tracks.push_back(new_track);
        }
    }
    
    // 更新轨迹列表
    tracked_tracks_.clear();
    lost_tracks_.clear();
    
    for (auto& track : activated_tracks) {
        if (track->state == TrackState::Tracked || track->state == TrackState::New) {
            tracked_tracks_.push_back(track);
        }
    }
    
    for (auto& track : refound_tracks) {
        track->state = TrackState::Tracked;
        tracked_tracks_.push_back(track);
    }
    
    for (auto& track : unmatched_tracks) {
        if (track->state == TrackState::Lost && track->time_since_update <= max_time_lost_) {
            lost_tracks_.push_back(track);
        }
    }
    
    // 构建输出结果
    std::vector<TrackResult> results;
    for (const auto& track : tracked_tracks_) {
        if (track->state == TrackState::Tracked) {
            TrackResult result;
            result.track_id = track->track_id;
            result.x1 = track->bbox[0];
            result.y1 = track->bbox[1];
            result.x2 = track->bbox[2];
            result.y2 = track->bbox[3];
            result.class_id = track->class_id;
            result.trajectory = track->trajectory;
            results.push_back(result);
        }
    }
    
    return results;
}

Eigen::MatrixXf ByteTracker::computeCostMatrix(
    const std::vector<std::shared_ptr<Track>>& tracks,
    const std::vector<const DetectionResult*>& detections) {
    
    int num_tracks = tracks.size();
    int num_dets = detections.size();
    
    Eigen::MatrixXf cost_matrix(num_tracks, num_dets);
    
    for (int i = 0; i < num_tracks; ++i) {
        for (int j = 0; j < num_dets; ++j) {
            float iou = computeIoU(tracks[i]->bbox, *detections[j]);
            cost_matrix(i, j) = 1.0f - iou;  // 转换为代价（IoU越大代价越小）
        }
    }
    
    return cost_matrix;
}

float ByteTracker::computeIoU(const float* bbox, const DetectionResult& det) {
    float x1 = std::max(bbox[0], det.x1);
    float y1 = std::max(bbox[1], det.y1);
    float x2 = std::min(bbox[2], det.x2);
    float y2 = std::min(bbox[3], det.y2);
    
    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;
    
    float area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
    float area2 = det.getArea();
    float union_area = area1 + area2 - inter_area;
    
    if (union_area <= 0) return 0.0f;
    
    return inter_area / union_area;
}

void ByteTracker::linearAssignment(
    const Eigen::MatrixXf& cost_matrix,
    float threshold,
    std::vector<std::pair<int, int>>& matches,
    std::vector<int>& unmatched_track_indices,
    std::vector<int>& unmatched_det_indices) {
    
    int num_tracks = cost_matrix.rows();
    int num_dets = cost_matrix.cols();
    
    if (num_tracks == 0 || num_dets == 0) {
        for (int i = 0; i < num_tracks; ++i) {
            unmatched_track_indices.push_back(i);
        }
        for (int j = 0; j < num_dets; ++j) {
            unmatched_det_indices.push_back(j);
        }
        return;
    }
    
    // 简单贪婪匹配
    std::vector<bool> track_matched(num_tracks, false);
    std::vector<bool> det_matched(num_dets, false);
    
    // 收集所有有效候选
    std::vector<std::tuple<float, int, int>> candidates;
    for (int i = 0; i < num_tracks; ++i) {
        for (int j = 0; j < num_dets; ++j) {
            if (cost_matrix(i, j) <= threshold) {
                candidates.push_back({cost_matrix(i, j), i, j});
            }
        }
    }
    
    // 按代价排序
    std::sort(candidates.begin(), candidates.end());
    
    // 贪婪选择
    for (const auto& [cost, track_idx, det_idx] : candidates) {
        if (!track_matched[track_idx] && !det_matched[det_idx]) {
            matches.push_back({track_idx, det_idx});
            track_matched[track_idx] = true;
            det_matched[det_idx] = true;
        }
    }
    
    // 收集未匹配项
    for (int i = 0; i < num_tracks; ++i) {
        if (!track_matched[i]) {
            unmatched_track_indices.push_back(i);
        }
    }
    for (int j = 0; j < num_dets; ++j) {
        if (!det_matched[j]) {
            unmatched_det_indices.push_back(j);
        }
    }
}

} // namespace infrared
