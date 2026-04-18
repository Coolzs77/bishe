// ========================================================================
//  板端红外视频检测程序
// ========================================================================
//
// 用法:
//   # 红外视频文件检测
//   ./bishe_rknn_video <model.rknn> <红外视频.mp4> [标签文件] [输出视频] [conf] [nms] [track]
//
//   # 红外摄像头实时检测 (后期接上红外摄像头时使用)
//   ./bishe_rknn_video <model.rknn> /dev/video0 [标签文件] [输出视频] [conf] [nms] [track]
//   ./bishe_rknn_video <model.rknn> camera:0    [标签文件] [输出视频] [conf] [nms] [track]
//
// 说明:
//   本程序逐帧读取红外热成像视频, 使用 RKNN NPU 执行 YOLOv5 推理,
//   检测 person 和 car 两类目标, 在每帧上绘制检测框,
//   并把结果写入输出视频文件.
//
//   依赖 OpenCV 进行视频读写. RV1126B SDK 通常自带 OpenCV.
//   如果板子没有 OpenCV, 需要先交叉编译安装.
//
//   红外视频帧虽然看起来是灰度的, 但 VideoCapture 默认解码为
//   BGR 三通道, 和训练时 OpenCV 的行为一致.
// ========================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include "file_utils.h"
#include "image_utils.h"
#include "rknn_detector.hpp"

namespace {

// ---- 计时工具 ----
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ---- 标签加载 ----
std::vector<std::string> load_labels(const char* path) {
    std::vector<std::string> labels;
    int line_count = 0;
    char** lines = read_lines_from_file(path, &line_count);
    if (lines == NULL) return labels;

    auto trim_ascii = [](std::string value) -> std::string {
        if (value.size() >= 3 &&
            static_cast<unsigned char>(value[0]) == 0xEF &&
            static_cast<unsigned char>(value[1]) == 0xBB &&
            static_cast<unsigned char>(value[2]) == 0xBF) {
            value = value.substr(3);
        }
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
            value.pop_back();
        }
        std::size_t first = 0;
        while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) {
            ++first;
        }
        return value.substr(first);
    };

    for (int i = 0; i < line_count; ++i) {
        if (lines[i] == NULL) continue;
        std::string cleaned = trim_ascii(lines[i]);
        if (!cleaned.empty()) labels.push_back(cleaned);
    }
    free_lines(lines, line_count);
    return labels;
}

const char* class_name(const std::vector<std::string>& labels, int id) {
    if (id < 0 || id >= static_cast<int>(labels.size())) return "unknown";
    return labels[id].c_str();
}

// ---- 颜色表 (BGR 格式, 用于 cv::Scalar) ----
cv::Scalar color_for_class(int class_id) {
    static const cv::Scalar colors[] = {
        cv::Scalar(255, 100, 50),   // person: 蓝色系
        cv::Scalar(50, 50, 255),    // car: 红色系
        cv::Scalar(50, 255, 50),    // 绿
        cv::Scalar(0, 255, 255),    // 黄
        cv::Scalar(0, 165, 255),    // 橙
        cv::Scalar(255, 255, 255),  // 白
    };
    return colors[class_id % 6];
}

// ---- cv::Mat → image_buffer_t 转换 (动态分配版, 调用者负责释放 out->virt_addr) ----
int mat_to_image_buffer(const cv::Mat& frame, image_buffer_t* out) {
    if (frame.empty() || out == NULL) return -1;

    cv::Mat rgb;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    } else if (frame.channels() == 1) {
        cv::cvtColor(frame, rgb, cv::COLOR_GRAY2RGB);
    } else {
        return -1;
    }

    memset(out, 0, sizeof(*out));
    out->width = rgb.cols;
    out->height = rgb.rows;
    out->format = IMAGE_FORMAT_RGB888;
    out->size = rgb.total() * rgb.elemSize();
    out->virt_addr = (unsigned char*)malloc(out->size);
    if (out->virt_addr == NULL) return -1;
    memcpy(out->virt_addr, rgb.data, out->size);
    return 0;
}

// ---- cv::Mat → image_buffer_t 转换 (复用外部缓冲区, 无 malloc) ----
// buf 必须已分配至少 frame.cols * frame.rows * 3 字节.
// out->virt_addr 指向 buf, 调用者不需要释放.
int mat_to_image_buffer_inplace(const cv::Mat& frame, image_buffer_t* out, unsigned char* buf) {
    if (frame.empty() || out == NULL || buf == NULL) return -1;

    cv::Mat rgb(frame.rows, frame.cols, CV_8UC3, buf);
    if (frame.channels() == 3) {
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    } else if (frame.channels() == 1) {
        cv::cvtColor(frame, rgb, cv::COLOR_GRAY2RGB);
    } else {
        return -1;
    }

    memset(out, 0, sizeof(*out));
    out->width = frame.cols;
    out->height = frame.rows;
    out->format = IMAGE_FORMAT_RGB888;
    out->size = frame.cols * frame.rows * 3;
    out->virt_addr = buf;
    return 0;
}

// ---- OpenCV NEON Letterbox 预处理 (BGR→RGB + resize + pad) ----
//
// 将任意尺寸的 BGR 帧一步转换为模型输入格式 (model_w×model_h×3 RGB888).
// 完全使用 OpenCV NEON 加速，预期耗时 2~3ms（vs. RGA/CPU letterbox 的 8~10ms）.
//
// dst_buf: app_ctx->input_buffer (model_w * model_h * 3 字节, 已预分配)
// tmp_buf: frame_rgb_buf (frame_w * frame_h * 3 字节, 已预分配, 用作 resize 临时缓冲区)
//
// 返回 letterbox 参数，用于检测坐标反映射.
letterbox_t preprocess_to_model_buf(
    const cv::Mat& frame,
    unsigned char* dst_buf,
    unsigned char* tmp_buf,
    int model_w,
    int model_h) {
    const float scale = std::min(static_cast<float>(model_w) / frame.cols,
                                 static_cast<float>(model_h) / frame.rows);
    const int new_w = std::max(1, static_cast<int>(roundf(frame.cols * scale)));
    const int new_h = std::max(1, static_cast<int>(roundf(frame.rows * scale)));
    const int left  = (model_w - new_w) / 2;
    const int top   = (model_h - new_h) / 2;

    // 1. 用灰色(114)填充整个模型输入缓冲区
    memset(dst_buf, 114, model_w * model_h * 3);

    // 2. Resize BGR frame → tmp_buf (NEON 加速, ~1.5ms)
    cv::Mat resized_bgr(new_h, new_w, CV_8UC3, tmp_buf);
    cv::resize(frame, resized_bgr, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 3. BGR→RGB 并写入 dst_buf 的 ROI (~0.5ms)
    cv::Mat model_mat(model_h, model_w, CV_8UC3, dst_buf);
    cv::Mat roi = model_mat(cv::Rect(left, top, new_w, new_h));
    cv::cvtColor(resized_bgr, roi, cv::COLOR_BGR2RGB);

    letterbox_t lb;
    lb.scale = scale;
    lb.x_pad = left;
    lb.y_pad = top;
    return lb;
}

// ---- 在 cv::Mat 上绘制检测结果 ----
void draw_detections_on_mat(
    cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::vector<std::string>& labels) {
    char text[128];
    for (size_t i = 0; i < detections.size(); ++i) {
        const Detection& det = detections[i];
        int x1 = static_cast<int>(det.x1);
        int y1 = static_cast<int>(det.y1);
        int x2 = static_cast<int>(det.x2);
        int y2 = static_cast<int>(det.y2);
        cv::Scalar color = color_for_class(det.class_id);

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
        snprintf(text, sizeof(text), "%s %.0f%%",
                 class_name(labels, det.class_id), det.score * 100.0f);

        // 文字背景框
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int text_y = (y1 > text_size.height + 4) ? y1 - 4 : y1 + text_size.height + 4;
        cv::rectangle(frame,
                      cv::Point(x1, text_y - text_size.height - 2),
                      cv::Point(x1 + text_size.width, text_y + 2),
                      color, cv::FILLED);
        cv::putText(frame, text, cv::Point(x1, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

// ---- 判断输入是否为摄像头设备 ----
bool is_camera_input(const char* path) {
    // camera:N 格式
    if (strncmp(path, "camera:", 7) == 0) return true;
    // /dev/videoN 格式
    if (strncmp(path, "/dev/video", 10) == 0) return true;
    return false;
}

int parse_camera_index(const char* path) {
    if (strncmp(path, "camera:", 7) == 0) {
        return atoi(path + 7);
    }
    if (strncmp(path, "/dev/video", 10) == 0) {
        return atoi(path + 10);
    }
    return 0;
}

bool parse_bool_arg(const char* value, bool default_value) {
    if (value == NULL) return default_value;
    if (strcmp(value, "1") == 0 || strcmp(value, "true") == 0 || strcmp(value, "on") == 0) return true;
    if (strcmp(value, "0") == 0 || strcmp(value, "false") == 0 || strcmp(value, "off") == 0) return false;
    return default_value;
}

// is_eiou_model_path 已不再用于自动切换阈值（已统一 conf/nms），
// 保留函数以备将来扩展或调试打印使用。
bool is_eiou_model_path(const char* model_path) {
    if (model_path == NULL) return false;
    std::string lower(model_path);
    for (std::size_t i = 0; i < lower.size(); ++i) {
        if (lower[i] >= 'A' && lower[i] <= 'Z') {
            lower[i] = static_cast<char>(lower[i] - 'A' + 'a');
        }
    }
    return lower.find("eiou") != std::string::npos;
}

float compute_iou(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}

struct TemporalTrack {
    int track_id;
    int class_id;
    int hits;
    int time_since_update;
    float score;
    Detection box;
};

class ByteTrackAlignTracker {
public:
    ByteTrackAlignTracker(
        int max_age,
        int min_hits,
        float iou_threshold,
        float high_threshold,
        float low_threshold,
        float match_iou_threshold,
        float second_match_iou_threshold,
        float reactivate_iou_threshold,
        int visible_lag)
        : max_age_(max_age),
          min_hits_(min_hits),
          high_threshold_(high_threshold),
          low_threshold_(low_threshold),
          match_iou_threshold_(match_iou_threshold),
          second_match_iou_threshold_(second_match_iou_threshold),
          reactivate_iou_threshold_(reactivate_iou_threshold),
          visible_lag_(visible_lag),
          lost_track_buffer_(max_age * 2),
          next_track_id_(1),
          frame_count_(0) {
        (void)iou_threshold;
    }

    std::vector<TemporalTrack> update(const std::vector<Detection>& detections) {
        ++frame_count_;

        for (std::size_t i = 0; i < active_tracks_.size(); ++i) {
            predict_track(*active_tracks_[i]);
        }
        for (std::size_t i = 0; i < lost_tracks_.size(); ++i) {
            predict_track(*lost_tracks_[i]);
        }

        std::vector<int> hi_det_indices;
        std::vector<int> lo_det_indices;
        hi_det_indices.reserve(detections.size());
        lo_det_indices.reserve(detections.size());
        for (std::size_t i = 0; i < detections.size(); ++i) {
            if (detections[i].score >= high_threshold_) {
                hi_det_indices.push_back(static_cast<int>(i));
            } else if (detections[i].score >= low_threshold_) {
                lo_det_indices.push_back(static_cast<int>(i));
            }
        }

        std::vector<std::pair<int, int> > matched_a;
        std::vector<int> unmatched_active;
        std::vector<int> unmatched_hi;
        associate(active_tracks_, detections, hi_det_indices, match_iou_threshold_, matched_a, unmatched_active, unmatched_hi);
        apply_matches(active_tracks_, detections, matched_a, hi_det_indices);

        if (!unmatched_active.empty() && !lo_det_indices.empty()) {
            std::vector<std::shared_ptr<TrackNode> > candidate_tracks;
            candidate_tracks.reserve(unmatched_active.size());
            for (std::size_t i = 0; i < unmatched_active.size(); ++i) {
                candidate_tracks.push_back(active_tracks_[unmatched_active[i]]);
            }
            std::vector<std::pair<int, int> > matched_b;
            std::vector<int> unmatched_active_rel;
            std::vector<int> unmatched_lo_ignored;
            associate(candidate_tracks, detections, lo_det_indices, second_match_iou_threshold_,
                      matched_b, unmatched_active_rel, unmatched_lo_ignored);
            for (std::size_t i = 0; i < matched_b.size(); ++i) {
                const int global_track_idx = unmatched_active[matched_b[i].first];
                const int lo_pos = matched_b[i].second;
                const int det_idx = lo_det_indices[lo_pos];
                update_track(*active_tracks_[global_track_idx], detections[det_idx]);
            }
            std::vector<int> unresolved_active;
            unresolved_active.reserve(unmatched_active_rel.size());
            for (std::size_t i = 0; i < unmatched_active_rel.size(); ++i) {
                unresolved_active.push_back(unmatched_active[unmatched_active_rel[i]]);
            }
            unmatched_active.swap(unresolved_active);
        }

        std::vector<int> remaining_hi_det_indices;
        remaining_hi_det_indices.reserve(unmatched_hi.size());
        for (std::size_t i = 0; i < unmatched_hi.size(); ++i) {
            remaining_hi_det_indices.push_back(hi_det_indices[unmatched_hi[i]]);
        }

        if (!lost_tracks_.empty() && !remaining_hi_det_indices.empty()) {
            std::vector<std::pair<int, int> > matched_lost;
            std::vector<int> unmatched_lost;
            std::vector<int> unmatched_rem;
            associate(lost_tracks_, detections, remaining_hi_det_indices, reactivate_iou_threshold_,
                      matched_lost, unmatched_lost, unmatched_rem);
            apply_matches(lost_tracks_, detections, matched_lost, remaining_hi_det_indices);

            std::set<int> reactivated_set;
            for (std::size_t i = 0; i < matched_lost.size(); ++i) {
                int idx = matched_lost[i].first;
                lost_tracks_[idx]->state = "confirmed";
                reactivated_set.insert(idx);
            }
            for (std::set<int>::iterator it = reactivated_set.begin(); it != reactivated_set.end(); ++it) {
                active_tracks_.push_back(lost_tracks_[*it]);
            }

            std::vector<std::shared_ptr<TrackNode> > next_lost;
            next_lost.reserve(unmatched_lost.size());
            for (std::size_t i = 0; i < unmatched_lost.size(); ++i) {
                next_lost.push_back(lost_tracks_[unmatched_lost[i]]);
            }
            lost_tracks_.swap(next_lost);

            std::vector<int> next_remaining_hi;
            next_remaining_hi.reserve(unmatched_rem.size());
            for (std::size_t i = 0; i < unmatched_rem.size(); ++i) {
                next_remaining_hi.push_back(remaining_hi_det_indices[unmatched_rem[i]]);
            }
            remaining_hi_det_indices.swap(next_remaining_hi);
        }

        handle_unmatched_active(unmatched_active);
        for (std::size_t i = 0; i < remaining_hi_det_indices.size(); ++i) {
            create_new_track(detections[remaining_hi_det_indices[i]]);
        }
        prune_lost_tracks();

        std::vector<TemporalTrack> outputs;
        outputs.reserve(active_tracks_.size());
        for (std::size_t i = 0; i < active_tracks_.size(); ++i) {
            const TrackNode& t = *active_tracks_[i];
            // confirmed 轨迹在 visible_lag 帧内保显（KF 预测平滑位置）
            if (t.state == "confirmed" && t.time_since_update <= visible_lag_) {
                outputs.push_back(to_temporal_track(t));
            } else if (frame_count_ <= min_hits_ && t.time_since_update == 0) {
                // 视频开始的前几帧，还没有 confirmed 轨迹，允许当帧匹配的轨迹直接上屏
                outputs.push_back(to_temporal_track(t));
            }
        }
        return outputs;
    }

private:
    struct TrackNode {
        cv::KalmanFilter kf;
        cv::Mat mean;
        int track_id;
        int class_id;
        float confidence;
        int hits;
        int age;
        int time_since_update;
        std::string state;
        std::map<int, int> class_votes;
        float smooth_w;   // 平滑后的宽度（仅用于显示，避免 a*h 乘法放大抖动）
        float smooth_h;   // 平滑后的高度
    };

    static cv::Mat make_measurement(const Detection& det) {
        const float w = std::max(1e-6f, det.x2 - det.x1);
        const float h = std::max(1e-6f, det.y2 - det.y1);
        cv::Mat z(4, 1, CV_32F);
        z.at<float>(0, 0) = (det.x1 + det.x2) * 0.5f;
        z.at<float>(1, 0) = (det.y1 + det.y2) * 0.5f;
        z.at<float>(2, 0) = w / h;
        z.at<float>(3, 0) = h;
        return z;
    }

    static Detection mean_to_detection(const cv::Mat& mean, int class_id, float score) {
        const float cx = mean.at<float>(0, 0);
        const float cy = mean.at<float>(1, 0);
        const float a = mean.at<float>(2, 0);
        const float h = std::max(1e-6f, mean.at<float>(3, 0));
        const float w = std::max(1e-6f, a * h);
        Detection det;
        det.x1 = cx - w * 0.5f;
        det.y1 = cy - h * 0.5f;
        det.x2 = cx + w * 0.5f;
        det.y2 = cy + h * 0.5f;
        det.class_id = class_id;
        det.score = score;
        return det;
    }

    void init_kf(TrackNode& node, const Detection& det) {
        node.kf.init(8, 4, 0, CV_32F);
        cv::setIdentity(node.kf.transitionMatrix);
        for (int i = 0; i < 4; ++i) {
            node.kf.transitionMatrix.at<float>(i, 4 + i) = 1.0f;
        }
        node.kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
        for (int i = 0; i < 4; ++i) {
            node.kf.measurementMatrix.at<float>(i, i) = 1.0f;
        }
        const cv::Mat z = make_measurement(det);
        node.kf.statePost = cv::Mat::zeros(8, 1, CV_32F);
        for (int i = 0; i < 4; ++i) node.kf.statePost.at<float>(i, 0) = z.at<float>(i, 0);
        node.mean = node.kf.statePost.clone();
        // 初始误差协方差 P0（与 PC KalmanFilter.initiate() 对齐，基于目标高度归一化）
        // std_weight_position=1/20, std_weight_velocity=1/160, 初始倍率 x2/x10
        const float h0 = std::max(1.0f, z.at<float>(3, 0));
        const float p0 = 2.0f * STD_WEIGHT_POS_ * h0;
        const float v0 = 10.0f * STD_WEIGHT_VEL_ * h0;
        const float p_diag[8] = {
            p0 * p0, p0 * p0, 1e-4f, p0 * p0,
            v0 * v0, v0 * v0, 1e-10f, v0 * v0
        };
        node.kf.errorCovPost = cv::Mat::zeros(8, 8, CV_32F);
        for (int i = 0; i < 8; ++i) node.kf.errorCovPost.at<float>(i, i) = p_diag[i];
        // Q/R 在 predict_track/update_track 按帧高度动态设置，此处初始化为 0
        node.kf.processNoiseCov     = cv::Mat::zeros(8, 8, CV_32F);
        node.kf.measurementNoiseCov = cv::Mat::zeros(4, 4, CV_32F);
        node.smooth_w = std::max(1e-6f, det.x2 - det.x1);
        node.smooth_h = std::max(1e-6f, det.y2 - det.y1);
    }

    void predict_track(TrackNode& node) {
        // 每帧动态设置过程噪声 Q（与 PC KalmanFilter.predict() 对齐，基于当前 h 归一化）
        const float h = std::max(1.0f, node.mean.at<float>(3, 0));
        const float qp = STD_WEIGHT_POS_ * h;
        const float qv = STD_WEIGHT_VEL_ * h;
        const float q_diag[8] = {
            qp * qp, qp * qp, 1e-4f, qp * qp,
            qv * qv, qv * qv, 1e-10f, qv * qv
        };
        for (int i = 0; i < 8; ++i)
            node.kf.processNoiseCov.at<float>(i, i) = q_diag[i];
        cv::Mat pred = node.kf.predict();
        node.mean = pred.clone();
        node.age += 1;
        node.time_since_update += 1;
    }

    void update_track(TrackNode& node, const Detection& det) {
        cv::Mat z = make_measurement(det);
        // 观测噪声 R：与 PC 端 KalmanFilter.update() 对齐，使用 STD_WEIGHT_POS_ (1/20)。
        // R=Q 时 KF 对检测和预测给予均衡权重，修正值紧跟实际位置 → IoU 匹配可靠。
        // 之前用 1/10 (加大 R) 反而让 KF 响应慢→预测偏离→IoU 下降→ID切换多。
        const float h = std::max(1.0f, z.at<float>(3, 0));
        const float rp = STD_WEIGHT_POS_ * h;
        const float r_diag[4] = {rp * rp, rp * rp, 1e-2f, rp * rp};
        for (int i = 0; i < 4; ++i)
            node.kf.measurementNoiseCov.at<float>(i, i) = r_diag[i];
        cv::Mat corr = node.kf.correct(z);
        node.mean = corr.clone();
        node.hits += 1;
        node.time_since_update = 0;

        const int cls = det.class_id;
        node.class_votes[cls] = node.class_votes[cls] + 1;
        int best_cls = node.class_id;
        int best_vote = -1;
        for (std::map<int, int>::const_iterator it = node.class_votes.begin(); it != node.class_votes.end(); ++it) {
            if (it->second > best_vote) {
                best_vote = it->second;
                best_cls = it->first;
            }
        }
        node.class_id = best_cls;
        node.confidence = 0.7f * det.score + 0.3f * node.confidence;
        // 从 KF 修正后的 mean 中提取尺寸，做轻量 EMA 平滑
        // KF 状态空间是 (cx,cy,a,h)，width=a*h 乘法会放大微小波动；
        // 对 w/h 单独做 EMA 可消除帧间尺寸抖动，而不影响位置跟踪。
        {
            const float kf_a = node.mean.at<float>(2, 0);
            const float kf_h = std::max(1e-6f, node.mean.at<float>(3, 0));
            const float kf_w = std::max(1e-6f, kf_a * kf_h);
            node.smooth_w = 0.3f * node.smooth_w + 0.7f * kf_w;
            node.smooth_h = 0.3f * node.smooth_h + 0.7f * kf_h;
        }
        if (node.state == "tentative" && node.hits >= min_hits_) {
            node.state = "confirmed";
        } else if (node.state == "lost") {
            node.state = "confirmed";
        }
    }

    TemporalTrack to_temporal_track(const TrackNode& node) const {
        TemporalTrack out;
        out.track_id = node.track_id;
        out.class_id = node.class_id;
        out.hits = node.hits;
        out.time_since_update = node.time_since_update;
        out.score = node.confidence;
        // 位置(cx,cy)：直接取 KF mean，响应快、无漂移
        // 尺寸(w,h)：取 smooth_w/smooth_h，消除 a*h 乘法抖动
        const float cx = node.mean.at<float>(0, 0);
        const float cy = node.mean.at<float>(1, 0);
        out.box.x1 = cx - node.smooth_w * 0.5f;
        out.box.y1 = cy - node.smooth_h * 0.5f;
        out.box.x2 = cx + node.smooth_w * 0.5f;
        out.box.y2 = cy + node.smooth_h * 0.5f;
        out.box.class_id = node.class_id;
        out.box.score = node.confidence;
        return out;
    }

    void associate(
        const std::vector<std::shared_ptr<TrackNode> >& tracks,
        const std::vector<Detection>& detections,
        const std::vector<int>& det_indices,
        float iou_thresh,
        std::vector<std::pair<int, int> >& matched,
        std::vector<int>& unmatched_tracks,
        std::vector<int>& unmatched_dets) {
        matched.clear();
        unmatched_tracks.clear();
        unmatched_dets.clear();
        if (tracks.empty()) {
            for (std::size_t j = 0; j < det_indices.size(); ++j) unmatched_dets.push_back(static_cast<int>(j));
            return;
        }
        if (det_indices.empty()) {
            for (std::size_t i = 0; i < tracks.size(); ++i) unmatched_tracks.push_back(static_cast<int>(i));
            return;
        }

        std::vector<std::tuple<float, int, int> > candidates;
        candidates.reserve(tracks.size() * det_indices.size());
        // 纯 IoU 匹配：与 PC 端 unified_tracker.py 完全对齐。
        // 当 KF 噪声参数 R=Q (1/20) 时，KF 修正值紧跟检测，预测框准确，IoU 足够可靠。
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            Detection tbox = mean_to_detection(tracks[i]->mean, tracks[i]->class_id, tracks[i]->confidence);
            for (std::size_t j = 0; j < det_indices.size(); ++j) {
                const Detection& det = detections[det_indices[j]];
                if (tracks[i]->class_id != det.class_id) continue;
                const float iou = compute_iou(tbox, det);
                if (iou < iou_thresh) continue;
                const float cost = 1.0f - iou;
                candidates.push_back(std::make_tuple(cost, static_cast<int>(i), static_cast<int>(j)));
            }
        }
        std::sort(candidates.begin(), candidates.end());
        std::vector<int> track_used(tracks.size(), 0);
        std::vector<int> det_used(det_indices.size(), 0);
        for (std::size_t k = 0; k < candidates.size(); ++k) {
            const int ti = std::get<1>(candidates[k]);
            const int dj = std::get<2>(candidates[k]);
            if (track_used[ti] || det_used[dj]) continue;
            track_used[ti] = 1;
            det_used[dj] = 1;
            matched.push_back(std::make_pair(ti, dj));
        }
        for (std::size_t i = 0; i < tracks.size(); ++i) if (!track_used[i]) unmatched_tracks.push_back(static_cast<int>(i));
        for (std::size_t j = 0; j < det_indices.size(); ++j) if (!det_used[j]) unmatched_dets.push_back(static_cast<int>(j));
    }

    void apply_matches(
        std::vector<std::shared_ptr<TrackNode> >& tracks,
        const std::vector<Detection>& detections,
        const std::vector<std::pair<int, int> >& matched,
        const std::vector<int>& det_indices) {
        for (std::size_t i = 0; i < matched.size(); ++i) {
            const int ti = matched[i].first;
            const int det_pos = matched[i].second;
            update_track(*tracks[ti], detections[det_indices[det_pos]]);
        }
    }

    void handle_unmatched_active(const std::vector<int>& unmatched_active) {
        if (unmatched_active.empty()) return;
        std::set<int> unmatched(unmatched_active.begin(), unmatched_active.end());
        std::vector<std::shared_ptr<TrackNode> > keep_active;
        std::vector<std::shared_ptr<TrackNode> > move_lost;
        keep_active.reserve(active_tracks_.size());
        for (std::size_t i = 0; i < active_tracks_.size(); ++i) {
            std::shared_ptr<TrackNode> t = active_tracks_[i];
            if (unmatched.count(static_cast<int>(i))) {
                // tentative 轨迹：给 min_hits 帧宽限（与确认窗口对齐），
                // 超时后直接丢弃（不进 lost），避免噪声轨迹长期占坑抢匹配。
                // confirmed 轨迹：max_age 内保留 active，超时后移入 lost 等待重激活。
                if (t->state == "tentative") {
                    if (t->time_since_update > min_hits_) {
                        // 直接丢弃，不进 lost（噪声轨迹没有值得保留的历史）
                    } else {
                        keep_active.push_back(t);
                    }
                } else if (t->time_since_update > max_age_) {
                    t->state = "lost";
                    move_lost.push_back(t);
                } else {
                    keep_active.push_back(t);
                }
            } else {
                keep_active.push_back(t);
            }
        }
        active_tracks_.swap(keep_active);
        lost_tracks_.insert(lost_tracks_.end(), move_lost.begin(), move_lost.end());
    }

    void create_new_track(const Detection& det) {
        std::shared_ptr<TrackNode> node(new TrackNode());
        node->track_id = next_track_id_++;
        node->class_id = det.class_id;
        node->confidence = det.score;
        node->hits = 1;
        node->age = 1;
        node->time_since_update = 0;
        node->state = "tentative";
        node->class_votes.clear();
        node->class_votes[node->class_id] = 1;
        init_kf(*node, det);
        active_tracks_.push_back(node);
    }

    void prune_lost_tracks() {
        std::vector<std::shared_ptr<TrackNode> > kept;
        kept.reserve(lost_tracks_.size());
        for (std::size_t i = 0; i < lost_tracks_.size(); ++i) {
            if (lost_tracks_[i]->time_since_update <= lost_track_buffer_) {
                kept.push_back(lost_tracks_[i]);
            }
        }
        lost_tracks_.swap(kept);
    }

    // 卡尔曼滤波器噪声权重（与 PC 端 KalmanFilter 完全对齐）
    // STD_WEIGHT_POS = 1/20: 过程噪声 Q 和 观测噪声 R 共用同一权重
    // STD_WEIGHT_VEL = 1/160: 速度过程噪声
    static const float STD_WEIGHT_POS_;
    static const float STD_WEIGHT_VEL_;

    int max_age_;
    int min_hits_;
    float high_threshold_;
    float low_threshold_;
    float match_iou_threshold_;
    float second_match_iou_threshold_;
    float reactivate_iou_threshold_;
    int visible_lag_;
    int lost_track_buffer_;
    int next_track_id_;
    int frame_count_;
    std::vector<std::shared_ptr<TrackNode> > active_tracks_;
    std::vector<std::shared_ptr<TrackNode> > lost_tracks_;
};

const float ByteTrackAlignTracker::STD_WEIGHT_POS_  = 1.0f / 20.0f;
const float ByteTrackAlignTracker::STD_WEIGHT_VEL_  = 1.0f / 160.0f;

cv::Scalar color_for_track_id(int track_id) {
    static std::vector<cv::Scalar> palette;
    if (palette.empty()) {
        cv::RNG rng(42);
        palette.reserve(200);
        for (int i = 0; i < 200; ++i) {
            palette.push_back(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
        }
    }
    return palette[track_id % static_cast<int>(palette.size())];
}

void draw_hud_chip(
    cv::Mat& img,
    const std::string& text,
    int x,
    int y,
    const cv::Scalar& text_color,
    const cv::Scalar& bg_color,
    const cv::Scalar& accent_color,
    double alpha = 0.36) {
    const int pad_x = 10;
    const int pad_y = 8;
    const int radius = 8;
    const int thickness = 1;
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = 0.62;
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, font, font_scale, thickness, &baseline);
    int w = text_size.width + pad_x * 2;
    int h = text_size.height + pad_y * 2;

    int x1 = std::max(0, std::min(x, img.cols - 2));
    int y1 = std::max(0, std::min(y, img.rows - 2));
    int x2 = std::max(x1 + 1, std::min(x + w, img.cols - 1));
    int y2 = std::max(y1 + 1, std::min(y + h, img.rows - 1));
    int r = std::min(radius, std::min((x2 - x1) / 2, (y2 - y1) / 2));

    // 只对芯片区域做 alpha blend，避免全帧 clone() 开销
    cv::Rect chip_rect(x1, y1, x2 - x1, y2 - y1);
    cv::Mat chip_overlay = img(chip_rect).clone();  // 仅复制芯片小区域
    cv::rectangle(chip_overlay, cv::Point(r, 0), cv::Point(x2-x1-r, y2-y1), bg_color, cv::FILLED);
    cv::rectangle(chip_overlay, cv::Point(0, r), cv::Point(x2-x1, y2-y1-r), bg_color, cv::FILLED);
    cv::circle(chip_overlay, cv::Point(r, r), r, bg_color, cv::FILLED);
    cv::circle(chip_overlay, cv::Point(x2-x1-r, r), r, bg_color, cv::FILLED);
    cv::circle(chip_overlay, cv::Point(r, y2-y1-r), r, bg_color, cv::FILLED);
    cv::circle(chip_overlay, cv::Point(x2-x1-r, y2-y1-r), r, bg_color, cv::FILLED);
    cv::Mat chip_dst = img(chip_rect);
    cv::addWeighted(chip_overlay, alpha, chip_dst, 1.0 - alpha, 0.0, chip_dst);
    cv::line(img, cv::Point(x1 + 5, y1 + 5), cv::Point(x1 + 5, y2 - 5), accent_color, 2, cv::LINE_AA);

    cv::Point org(x1 + pad_x, y2 - pad_y);
    cv::putText(img, text, org, font, font_scale, cv::Scalar(12, 14, 18), 2, cv::LINE_AA);
    cv::putText(img, text, org, font, font_scale, text_color, 1, cv::LINE_AA);
}

void draw_tracks_on_mat(
    cv::Mat& frame,
    const std::vector<TemporalTrack>& tracks,
    const std::vector<std::string>& labels) {
    if (tracks.empty()) return;

    // 仅复制一次全帧用于所有标签背景的 alpha blend，避免 N 次全帧 clone()
    cv::Mat overlay;
    frame.copyTo(overlay);

    for (size_t i = 0; i < tracks.size(); ++i) {
        const TemporalTrack& tr = tracks[i];
        int x1 = static_cast<int>(tr.box.x1);
        int y1 = static_cast<int>(tr.box.y1);
        int x2 = static_cast<int>(tr.box.x2);
        int y2 = static_cast<int>(tr.box.y2);
        int thickness = (tr.time_since_update == 0) ? 2 : 1;
        cv::Scalar color = color_for_track_id(tr.track_id);
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);

        char label[128];
        snprintf(label, sizeof(label), "ID:%d %s", tr.track_id, class_name(labels, tr.class_id));
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.48, 1, &baseline);
        int ly2 = (y1 > ts.height + 8) ? y1 : (y1 + ts.height + 8);
        int ly1 = ly2 - ts.height - 6;
        int lx1 = std::max(0, x1);
        int lx2 = std::min(frame.cols - 1, lx1 + ts.width + 8);

        // 在 overlay 上画标签背景矩形
        cv::rectangle(overlay, cv::Point(lx1, ly1), cv::Point(lx2, ly2), color, cv::FILLED);
    }

    // 一次 addWeighted 完成所有标签背景的 alpha blend
    cv::addWeighted(overlay, 0.45, frame, 0.55, 0, frame);

    // 在 blend 后的帧上画文字
    for (size_t i = 0; i < tracks.size(); ++i) {
        const TemporalTrack& tr = tracks[i];
        int x1 = static_cast<int>(tr.box.x1);
        int y1 = static_cast<int>(tr.box.y1);
        char label[128];
        snprintf(label, sizeof(label), "ID:%d %s", tr.track_id, class_name(labels, tr.class_id));
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.48, 1, &baseline);
        int ly2 = (y1 > ts.height + 8) ? y1 : (y1 + ts.height + 8);
        int lx1 = std::max(0, x1);
        cv::putText(frame, label, cv::Point(lx1 + 4, ly2 - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
        cv::putText(frame, label, cv::Point(lx1 + 4, ly2 - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("红外视频检测 - 板端推理程序\n\n");
        printf("用法:\n");
        printf("  视频文件: %s <model.rknn> <视频.mp4> [标签] [输出视频|-] [conf] [nms] [track] [overlay]\n", argv[0]);
        printf("  摄像头:   %s <model.rknn> camera:0   [标签] [输出视频|-] [conf] [nms] [track] [overlay]\n", argv[0]);
        printf("  摄像头:   %s <model.rknn> /dev/video0 [标签] [输出视频|-] [conf] [nms] [track] [overlay]\n", argv[0]);
        printf("  track:   1/0 或 on/off (默认 1, 与 PC 检测+跟踪链路对齐)\n");
        printf("  overlay: 1/0 或 on/off (默认 1; 设为 0 跳过绘制提高计算帧率)\n");
        printf("  输出视频为 - 或 none 时跳过磁盘写入，仅计算 FPS\n");
        return -1;
    }

    const char* model_path  = argv[1];
    const char* input_path  = argv[2];
    const char* labels_path = argc >= 4 ? argv[3] : "./model/infrared_labels.txt";
    const char* output_path = argc >= 5 ? argv[4] : "out_video.mp4";
    // 统一默认阈值: conf=0.25 / nms=0.45，与 PC 端评估一致，不区分模型类型。
    // 若需覆盖请通过第 5/6 参数传入: ./bishe_rknn_video model.rknn video.mp4 labels out 0.55 0.35
    const float DEFAULT_CONF = 0.25f;
    const float DEFAULT_NMS  = 0.45f;
    float conf_threshold    = argc >= 6 ? static_cast<float>(atof(argv[5])) : DEFAULT_CONF;
    float nms_threshold     = argc >= 7 ? static_cast<float>(atof(argv[6])) : DEFAULT_NMS;
    bool enable_track_align = argc >= 8 ? parse_bool_arg(argv[7], true) : true;
    bool enable_overlay     = argc >= 9 ? parse_bool_arg(argv[8], true) : true;

    // 输出路径为 "-" 或 "none" 时跳过磁盘写入，用于纯计算 FPS 测试
    bool save_video = (output_path != NULL &&
                       strcmp(output_path, "-") != 0 &&
                       strcmp(output_path, "none") != 0);

    printf("=== 红外视频检测 ===\n");
    printf("  模型:   %s\n", model_path);
    printf("  输入:   %s\n", input_path);
    printf("  标签:   %s\n", labels_path);
    printf("  输出:   %s%s\n", output_path, save_video ? "" : "  (跳过写入，纯计算模式)");
    printf("  conf:   %.2f\n", conf_threshold);
    printf("  nms:    %.2f\n", nms_threshold);
    printf("  track:  %s\n", enable_track_align ? "on" : "off");
    printf("  overlay:%s\n", enable_overlay ? "on" : "off (skip draw)");
    if (argc < 6) {
        printf("  note:   默认 conf=0.25, nms=0.45 (与 PC 评估一致; EIoU 模型如需更严格请传第5/6参数 0.55 0.35)\n");
    }
    printf("\n");

    // ---- 加载标签 ----
    std::vector<std::string> labels = load_labels(labels_path);
    if (labels.empty()) {
        printf("[ERROR] 加载标签文件失败: %s\n", labels_path);
        return -1;
    }

    // ---- 初始化 RKNN 模型 ----
    RknnAppContext app_ctx;
    if (init_rknn_model(model_path, &app_ctx) != 0) {
        printf("[ERROR] 初始化 RKNN 模型失败\n");
        return -1;
    }

    // ---- 打开视频源 ----
    cv::VideoCapture cap;
    bool from_camera = is_camera_input(input_path);

    if (from_camera) {
        int cam_idx = parse_camera_index(input_path);
        printf("打开红外摄像头: 设备 %d ...\n", cam_idx);
        cap.open(cam_idx);
    } else {
        printf("打开红外视频: %s ...\n", input_path);
        cap.open(input_path);
    }

    if (!cap.isOpened()) {
        printf("[ERROR] 无法打开视频源: %s\n", input_path);
        release_rknn_model(&app_ctx);
        return -1;
    }

    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps   = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    if (fps <= 0.0) fps = 25.0;
    printf("视频信息: %dx%d, %.1f FPS", frame_w, frame_h, fps);
    if (!from_camera && total_frames > 0)
        printf(", 共 %d 帧", total_frames);
    printf("\n\n");

    // ---- 创建输出视频 ----
    cv::VideoWriter writer;
    if (save_video) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(output_path, fourcc, fps, cv::Size(frame_w, frame_h), true);
        if (!writer.isOpened()) {
            printf("[WARN] 无法创建输出视频: %s (仅打印结果, 不保存视频)\n", output_path);
        }
    }

    // 预分配帧缓冲区，避免每帧 mat_to_image_buffer 内部 malloc/free
    int frame_buf_size = frame_w * frame_h * 3;
    unsigned char* frame_rgb_buf = (unsigned char*)malloc(frame_buf_size);
    if (frame_rgb_buf == NULL) {
        printf("[ERROR] 预分配帧缓冲区失败 (%d 字节)\n", frame_buf_size);
        cap.release();
        release_rknn_model(&app_ctx);
        return -1;
    }

    // ---- 逐帧推理循环 ----
    cv::Mat frame;
    int frame_count = 0;
    int total_detections = 0;      // NPU 原始检测数（NMS后）
    int total_track_displays = 0; // ByteTrack 实际绘制轨迹数
    double total_infer_ms = 0.0;
    double total_npu_ms = 0.0;
    double total_preprocess_ms = 0.0;
    double total_track_ms = 0.0;
    double loop_start = get_time_ms();
    // R2 迭代优化参数（基于板端 3 轮实验选定）:
    // high_threshold=0.5, low_threshold=0.1, match_iou=0.25, second_match_iou=0.15
    // max_age=30, min_hits=2, visible_lag=3, reactivate_iou=0.20
    // R2 vs 基线: seq006 唯一ID -14%, 轨迹展示 +18%; seq009 唯一ID -11%, 轨迹展示 +6%
    const float track_high_threshold = 0.50f;
    const float track_low_threshold = 0.10f;
    ByteTrackAlignTracker byte_tracker(
        30,              // max_age
        2,               // min_hits              (R2: 3→2, 加速确认)
        0.3f,            // iou_threshold (unused, kept for compat)
        track_high_threshold,
        track_low_threshold,
        0.25f,           // match_iou_threshold    (R1: 0.30→0.25, 宽松匹配)
        0.15f,           // second_match_iou_threshold (R1: 0.20→0.15)
        0.20f,           // reactivate_iou_threshold
        3                // visible_lag            (R2: 1→3, 延长展示)
    );
    std::set<int> unique_track_ids;
    double fps_ema = -1.0;
    const double fps_alpha = 0.12;

    while (cap.read(frame)) {
        // 预处理: OpenCV NEON letterbox (resize + pad + BGR→RGB), 直接写入 input_buffer
        // 比 convert_image_with_exact_letterbox (CPU/RGA) 快约 6~7ms
        double pre_t0 = get_time_ms();
        letterbox_t letterbox = preprocess_to_model_buf(
            frame, app_ctx.input_buffer, frame_rgb_buf,
            app_ctx.model_width, app_ctx.model_height);
        total_preprocess_ms += get_time_ms() - pre_t0;

        // RKNN 推理 (跳过内部 letterbox, input_buffer 已由上一步填充)
        double t0 = get_time_ms();
        std::vector<Detection> detections;
        int ret = inference_rknn_model_preloaded(
            &app_ctx, &letterbox, frame.cols, frame.rows,
            conf_threshold, nms_threshold, &detections);
        double infer_ms = get_time_ms() - t0;

        if (ret != 0) {
            printf("[WARN] 帧 %d 推理失败, 跳过\n", frame_count);
            ++frame_count;
            continue;
        }

        total_infer_ms += infer_ms;
        total_npu_ms += app_ctx.last_npu_ms;
        int display_count = static_cast<int>(detections.size());
        int det_count = static_cast<int>(detections.size());
        int track_count = 0;
        double track_t0 = get_time_ms();
        if (enable_track_align) {
            std::vector<TemporalTrack> tracks = byte_tracker.update(detections);
            const double track_ms = get_time_ms() - track_t0;
            total_track_ms += track_ms;
            display_count = static_cast<int>(tracks.size());
            track_count = display_count;
            if (infer_ms + track_ms > 1e-6) {
                double inst_fps = 1000.0 / (infer_ms + track_ms);
                fps_ema = (fps_ema < 0.0) ? inst_fps : ((1.0 - fps_alpha) * fps_ema + fps_alpha * inst_fps);
            }
            total_detections += det_count;        // 只累计 NPU 原始检测数
            total_track_displays += track_count;   // 单独累计轨迹展示数
            for (std::size_t i = 0; i < tracks.size(); ++i) {
                unique_track_ids.insert(tracks[i].track_id);
            }
            if (enable_overlay) draw_tracks_on_mat(frame, tracks, labels);
        } else {
            total_track_ms += get_time_ms() - track_t0;
            total_detections += det_count;
            if (enable_overlay) draw_detections_on_mat(frame, detections, labels);
        }

        if (enable_overlay) {
            if (enable_track_align && fps_ema > 0.0) {
                char line1[128];
                snprintf(line1, sizeof(line1), "Frame %d   Det %d   Track %d   IDs %d",
                         frame_count, det_count, track_count, static_cast<int>(unique_track_ids.size()));
                draw_hud_chip(frame, line1, 12, 10,
                              cv::Scalar(223, 235, 255),
                              cv::Scalar(24, 31, 43),
                              cv::Scalar(104, 232, 255),
                              0.35);
                char fps_text[32];
                snprintf(fps_text, sizeof(fps_text), "FPS %.1f", fps_ema);
                int base = 0;
                int fps_w = cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.62, 1, &base).width + 20;
                int fps_x = frame.cols - fps_w - 12;
                draw_hud_chip(frame, fps_text, fps_x, 10,
                              cv::Scalar(255, 245, 214),
                              cv::Scalar(29, 37, 52),
                              cv::Scalar(255, 166, 108),
                              0.38);
            }
            if (!enable_track_align) {
                char info[128];
                snprintf(info, sizeof(info), "F:%d Det:%d %.1fms",
                         frame_count, display_count, infer_ms);
                cv::putText(frame, info, cv::Point(10, 25),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
        }

        // 写入输出视频
        if (save_video && writer.isOpened()) {
            writer.write(frame);
        }

        ++frame_count;

        // 每 100 帧打印一次进度（含分段计时）
        if (frame_count % 100 == 0) {
            double avg_pre_ms  = total_preprocess_ms / frame_count;
            double avg_infer_ms = total_infer_ms / frame_count;
            double avg_npu_ms  = total_npu_ms / frame_count;
            double avg_track_ms = total_track_ms / frame_count;
            if (avg_npu_ms < 1e-6) avg_npu_ms = 1e-6;
            double compute_fps = 1000.0 / (avg_infer_ms + avg_track_ms);
            if (!from_camera && total_frames > 0) {
                printf("  进度: %d/%d 帧 (%.0f%%)  pre=%.1fms npu=%.1fms infer=%.1fms track=%.1fms  计算FPS=%.1f\n",
                       frame_count, total_frames,
                       100.0 * frame_count / total_frames,
                       avg_pre_ms, avg_npu_ms, avg_infer_ms, avg_track_ms, compute_fps);
            } else {
                printf("  已处理: %d 帧  pre=%.1fms npu=%.1fms infer=%.1fms track=%.1fms  计算FPS=%.1f\n",
                       frame_count, avg_pre_ms, avg_npu_ms, avg_infer_ms, avg_track_ms, compute_fps);
            }
            fflush(stdout);  // adb shell/SSH 管道为全缓冲，必须主动 flush 才能实时看到进度
        }
    }

    double total_ms = get_time_ms() - loop_start;

    // ---- 汇总输出 ----
    printf("\n=== 推理完成 ===\n");
    printf("  总帧数:          %d\n", frame_count);
    printf("  总检测数 (NPU):  %d\n", total_detections);
    if (enable_track_align && total_track_displays > 0) {
        printf("  轨迹展示总数:    %d\n", total_track_displays);
        printf("  唯一轨迹ID数:    %d  (ID 切换指标)\n", static_cast<int>(unique_track_ids.size()));
    }
    if (frame_count > 0) {
        double avg_pre_ms  = total_preprocess_ms / frame_count;
        double avg_npu_ms  = total_npu_ms / frame_count;
        double avg_infer_ms = total_infer_ms / frame_count;
        double avg_track_ms = total_track_ms / frame_count;
        if (avg_npu_ms < 1e-6) avg_npu_ms = 1e-6;
        double compute_fps  = 1000.0 / (avg_infer_ms + avg_track_ms);
        double npu_fps      = 1000.0 / avg_npu_ms;
        printf("  ---- 分段计时 (ms/帧) ----\n");
        printf("  预处理 (BGR→RGB):   %.1f ms\n", avg_pre_ms);
        printf("  推理 (含预处理+后处理): %.1f ms\n", avg_infer_ms);
        printf("    └─ 纯 NPU:        %.1f ms  (NPU FPS %.1f)\n", avg_npu_ms, npu_fps);
        printf("  跟踪 (ByteTrack):   %.1f ms\n", avg_track_ms);
        printf("  ---- 目标帧率 ----\n");
        printf("  计算 FPS (推理+跟踪): %.1f  %s\n",
               compute_fps, compute_fps >= 25.0 ? "[✓ >=25fps]" : "[✗ <25fps]");
        printf("  端到端 FPS (含读写): %.1f\n", 1000.0 * frame_count / total_ms);
    }
    if (save_video && writer.isOpened()) {
        printf("  输出视频:   %s\n", output_path);
    }

    // ---- 释放资源 ----
    free(frame_rgb_buf);
    cap.release();
    writer.release();
    release_rknn_model(&app_ctx);
    return 0;
}
