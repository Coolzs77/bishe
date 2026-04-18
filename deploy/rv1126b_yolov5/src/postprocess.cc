#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <cstdio>
#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif

// ========================================================================
//  内部辅助函数
// ========================================================================
namespace {

// sigmoid 激活函数, 3-branch 解码时使用.
inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// sigmoid 反函数 (logit), 用于 conf 阈值预过滤, 避免对背景格子调用 expf.
// logit(p) = ln(p/(1-p)).  p 必须在 (0,1) 内.
inline float logit_f(float p) {
    return logf(p / (1.0f - p));
}

// 计算单个检测框面积 (宽 × 高), 负值裁剪为 0.
float box_area(const Detection& det) {
    const float width = std::max(0.0f, det.x2 - det.x1);
    const float height = std::max(0.0f, det.y2 - det.y1);
    return width * height;
}

float intersection_area(const Detection& lhs, const Detection& rhs) {
    Detection inter;
    inter.x1 = std::max(lhs.x1, rhs.x1);
    inter.y1 = std::max(lhs.y1, rhs.y1);
    inter.x2 = std::min(lhs.x2, rhs.x2);
    inter.y2 = std::min(lhs.y2, rhs.y2);
    return box_area(inter);
}

// 计算两个框的交并比 (IoU = Intersection / Union).
// IoU 越大说明两个框重叠越多, NMS 越可能删除低分框.
float compute_iou(const Detection& lhs, const Detection& rhs) {
    const float inter_area = intersection_area(lhs, rhs);
    const float union_area = box_area(lhs) + box_area(rhs) - inter_area;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return inter_area / union_area;
}

inline bool is_finite(float value) {
    return std::isfinite(value);
}

// 过滤数值异常/极小框/极端长宽比框，减少量化噪声引起的杂框。
bool is_valid_box(const Detection& det) {
    if (!is_finite(det.x1) || !is_finite(det.y1) || !is_finite(det.x2) || !is_finite(det.y2) ||
        !is_finite(det.score)) {
        return false;
    }

    const float width = det.x2 - det.x1;
    const float height = det.y2 - det.y1;
    if (width <= 2.0f || height <= 2.0f) {
        return false;
    }
    if (width * height <= 16.0f) {
        return false;
    }

    const float aspect = (width > height) ? (width / height) : (height / width);
    if (aspect > 20.0f) {
        return false;
    }
    return true;
}

}  // namespace

// ========================================================================
//  decode_yolov5_output — 解码 RKNN 输出
// ========================================================================
//
// 本模型输出张量 [1, 25200, 7], 展平后即为 25200 行 × 7 列的浮点数组.
// 每行布局:
//   index  含义             取值范围      说明
//   ─────  ────             ────────      ────
//    [0]   cx               0 ~ 640       中心 x, 像素坐标 (Detect 头已解码)
//    [1]   cy               0 ~ 640       中心 y
//    [2]   w                > 0           宽度
//    [3]   h                > 0           高度
//    [4]   obj_conf         0 ~ 1         目标置信度 (已 sigmoid)
//    [5]   person_conf      0 ~ 1         person 类别得分 (已 sigmoid)
//    [6]   car_conf         0 ~ 1         car 类别得分 (已 sigmoid)
//
// 25200 个 proposal 来自三个检测头:
//   P3/8:   行 [0,      19200)   80×80 网格 × 3 锚框
//   P4/16:  行 [19200,  24000)   40×40 网格 × 3 锚框
//   P5/32:  行 [24000,  25200)   20×20 网格 × 3 锚框
//
std::vector<Detection> decode_yolov5_output(
    const float* raw_output,
    int num_proposals,
    int output_dim,
    float conf_threshold) {
    std::vector<Detection> detections;

    // —— 前置校验 ——
    if (raw_output == nullptr || output_dim <= 5 || num_proposals <= 0) {
        return detections;
    }

    // 运行时检查: num_proposals 和 output_dim 是否和模型常量一致.
    // 不一致时仍允许继续 (兼容其他尺寸), 但打印警告.
    if (num_proposals != MODEL_NUM_PROPOSALS || output_dim != MODEL_OUTPUT_DIM) {
        printf("[WARN] decode: expected %d×%d, got %d×%d\n",
               MODEL_NUM_PROPOSALS, MODEL_OUTPUT_DIM,
               num_proposals, output_dim);
    }

    // nc = 每行维度减去 (cx, cy, w, h, obj) = output_dim - 5.
    // 本模型 nc=2.
    const int num_classes = output_dim - 5;

    // —— 逐行遍历 25200 个 proposal ——
    for (int proposal_index = 0; proposal_index < num_proposals; ++proposal_index) {
        // row 指向当前 proposal 的起始地址.
        const float* row = raw_output + proposal_index * output_dim;

        // ① 先看 obj_conf (row[4]), 低于阈值直接跳过.
        //    绝大多数 proposal 会在这里被过滤掉 (背景区域 conf ≈ 0).
        const float object_confidence = row[4];
        if (object_confidence < conf_threshold) {
            continue;
        }

        // ② 在 row[5] ~ row[6] 中找得分最高的类别.
        //    对 2 类模型来说就是比较 person_conf 和 car_conf.
        int best_class_id = 0;
        float best_class_confidence = row[5];
        for (int class_index = 1; class_index < num_classes; ++class_index) {
            const float class_confidence = row[5 + class_index];
            if (class_confidence > best_class_confidence) {
                best_class_confidence = class_confidence;
                best_class_id = class_index;
            }
        }

        // ③ 最终得分 = obj_conf × best_cls_conf.
        //    YOLOv5 的联合得分公式; 再用同一阈值过滤一次.
        const float final_score = object_confidence * best_class_confidence;
        if (final_score < conf_threshold) {
            continue;
        }

        // ④ 取出已解码的中心坐标 + 宽高, 转换为 (x1,y1,x2,y2) 格式.
        //    此时坐标仍在 640×640 letterbox 空间, 后续 scale_detections 会
        //    映射回原始图像尺寸.
        const float center_x = row[0];
        const float center_y = row[1];
        const float width = row[2];
        const float height = row[3];

        if (!is_finite(center_x) || !is_finite(center_y) || !is_finite(width) || !is_finite(height)) {
            continue;
        }
        if (width <= 0.0f || height <= 0.0f) {
            continue;
        }

        Detection detection;
        detection.x1 = center_x - width * 0.5f;
        detection.y1 = center_y - height * 0.5f;
        detection.x2 = center_x + width * 0.5f;
        detection.y2 = center_y + height * 0.5f;
        detection.score = final_score;
        detection.class_id = best_class_id;
        if (is_valid_box(detection)) {
            detections.push_back(detection);
        }
    }

    return detections;
}

// ========================================================================
//  scale_detections — letterbox 坐标映射回原图
// ========================================================================
//
// 推理前原图经过 letterbox 缩放 + 灰边填充, 检测框坐标在 640×640 空间中.
// 这里做逆变换还原到原图像素坐标:
//   1. 减去 padding (x_pad / y_pad)
//   2. 除以缩放比 (scale)
//   3. clamp 到图像边界 [0, original_width/height]
//
void scale_detections(
    std::vector<Detection>& detections,
    const letterbox_t& letterbox,
    int original_width,
    int original_height) {
    for (std::size_t index = 0; index < detections.size(); ++index) {
        Detection& det = detections[index];

        // 逆变换: (letterbox坐标 - padding) / scale = 原图坐标
        det.x1 = (det.x1 - letterbox.x_pad) / letterbox.scale;
        det.y1 = (det.y1 - letterbox.y_pad) / letterbox.scale;
        det.x2 = (det.x2 - letterbox.x_pad) / letterbox.scale;
        det.y2 = (det.y2 - letterbox.y_pad) / letterbox.scale;

        // clamp: 确保坐标不越界
        det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(original_width)));
        det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(original_height)));
        det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(original_width)));
        det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(original_height)));
    }
}

// ========================================================================
//  apply_nms — 按类别 NMS
// ========================================================================
//
// 算法步骤:
//   1. 把所有候选框按 class_id 分组
//   2. 每个类别内部, 按 score 从高到低排序
//   3. 贪心保留: 从最高分开始, 保留该框,
//      然后删除与它 IoU ≥ iou_threshold 的所有低分框
//   4. 合并所有类别的保留框输出
//
std::vector<Detection> apply_nms(
    const std::vector<Detection>& detections,
    float iou_threshold) {
    if (detections.empty()) {
        return std::vector<Detection>();
    }

    if (iou_threshold < 0.05f) {
        iou_threshold = 0.05f;
    } else if (iou_threshold > 0.95f) {
        iou_threshold = 0.95f;
    }

    std::vector<Detection> sanitized;
    sanitized.reserve(detections.size());
    for (std::size_t index = 0; index < detections.size(); ++index) {
        if (is_valid_box(detections[index])) {
            sanitized.push_back(detections[index]);
        }
    }

    // step 1: 按类别分组, key=class_id, value=原数组下标列表.
    std::map<int, std::vector<int> > class_to_indices;
    for (std::size_t index = 0; index < sanitized.size(); ++index) {
        class_to_indices[sanitized[index].class_id].push_back(static_cast<int>(index));
    }

    std::vector<Detection> kept;
    const float contain_threshold = std::min(0.95f, std::max(0.80f, iou_threshold + 0.25f));
    for (std::map<int, std::vector<int> >::iterator it = class_to_indices.begin();
         it != class_to_indices.end(); ++it) {
        std::vector<int>& indices = it->second;

        // step 2: 按 score 降序排列.
        std::sort(indices.begin(), indices.end(),
                  [&sanitized](int lhs, int rhs) {
                      return sanitized[lhs].score > sanitized[rhs].score;
                  });

        // 每类最多保留前若干高分候选，避免大量低分噪声拖垮 NMS 效果。
        const std::size_t kMaxCandidatesPerClass = 300;
        if (indices.size() > kMaxCandidatesPerClass) {
            indices.resize(kMaxCandidatesPerClass);
        }

        // step 3: 贪心 NMS.
        std::vector<bool> removed(indices.size(), false);
        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (removed[i]) {
                continue;
            }
            // 当前最高分框保留.
            kept.push_back(sanitized[indices[i]]);
            // 与它重叠过高的同类低分框标记删除.
            for (std::size_t j = i + 1; j < indices.size(); ++j) {
                if (removed[j]) {
                    continue;
                }
                const Detection& keep_det = sanitized[indices[i]];
                const Detection& cand_det = sanitized[indices[j]];
                const float iou = compute_iou(keep_det, cand_det);
                const float inter = intersection_area(keep_det, cand_det);
                const float min_area = std::min(box_area(keep_det), box_area(cand_det));
                const float contain = (min_area > 1e-6f) ? (inter / min_area) : 0.0f;
                if (iou >= iou_threshold || contain >= contain_threshold) {
                    removed[j] = true;
                }
            }
        }
    }

    return kept;
}

// ========================================================================
//  dedup_detections — NMS 后跨类去重 + 同类近似框合并
// ========================================================================
//
// EIoU loss 训练的模型回归更精确, 但 INT8 量化后容易在同一目标上
// 产生多个 score 相近、坐标微偏的候选框, 标准 NMS 的 IoU 阈值
// 刚好卡在边界而漏删.
//
// 本函数在 NMS 之后做二次清扫:
//   1. 同类: IoU >= same_class_iou → 删低分框
//   2. 跨类: IoU >= cross_class_iou → 删低分框
// 均按 score 降序贪心.
//
std::vector<Detection> dedup_detections(
    const std::vector<Detection>& detections,
    float same_class_iou,
    float cross_class_iou) {
    if (detections.size() <= 1) {
        return detections;
    }

    // 按 score 降序排列的索引.
    std::vector<int> order(detections.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = static_cast<int>(i);
    }
    std::sort(order.begin(), order.end(),
              [&detections](int a, int b) {
                  return detections[a].score > detections[b].score;
              });

    std::vector<bool> removed(detections.size(), false);
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (removed[order[i]]) continue;
        const Detection& keep = detections[order[i]];
        for (std::size_t j = i + 1; j < order.size(); ++j) {
            if (removed[order[j]]) continue;
            const Detection& cand = detections[order[j]];
            const float iou = compute_iou(keep, cand);
            const bool same_cls = (keep.class_id == cand.class_id);
            const float thresh = same_cls ? same_class_iou : cross_class_iou;
            if (iou >= thresh) {
                removed[order[j]] = true;
            }
        }
    }

    std::vector<Detection> result;
    result.reserve(detections.size());
    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (!removed[i]) {
            result.push_back(detections[i]);
        }
    }
    return result;
}

// ========================================================================
//  decode_yolov5_3branch_output — 3-branch 原始输出解码
// ========================================================================
//
// 当 RKNN 模型生成 3 个输出张量 (对应 P3/P4/P5 三个检测头的 raw conv 输出)
// 时, 输出数据是未经 Detect 头解码的原始值, 需要:
//   1. sigmoid 激活全部属性
//   2. xy: (sigmoid(tx)*2 - 0.5 + grid_offset) * stride → 像素坐标
//   3. wh: (sigmoid(tw)*2)^2 * anchor → 像素尺寸
//   4. score = obj_conf * best_cls_conf, 低于阈值丢弃
//
std::vector<Detection> decode_yolov5_3branch_output(
    const BranchInfo branches[3],
    int num_anchors,
    int num_classes,
    float conf_threshold) {
    std::vector<Detection> detections;
    const int attr_count = 5 + num_classes;  // 本项目 = 7

    // ---- 预计算 logit(conf_threshold) ----
    // sigmoid 是单调函数, 因此:
    //   sigmoid(val[4]) >= conf_threshold  ⟺  val[4] >= logit(conf_threshold)
    // 绝大多数背景格子的 obj_logit 远低于阈值, 可在调用 expf 之前就跳过,
    // 将 sigmoid 调用量从 ~176K 次降至 ~2K 次 (取决于场景), 节省约 7ms CPU.
    const float obj_logit_thresh = logit_f(conf_threshold);
    // 类别得分预过滤阈值: score = obj_conf * cls_conf >= conf_threshold
    // 最宽松的情况: obj_conf=1 → cls_conf >= conf_threshold → cls_logit >= obj_logit_thresh
    const float cls_logit_thresh = obj_logit_thresh;

    for (int branch_idx = 0; branch_idx < 3; ++branch_idx) {
        const BranchInfo& info = branches[branch_idx];
        const int grid_h = info.grid_h;
        const int grid_w = info.grid_w;
        const int stride = info.stride;
        const int channel = num_anchors * attr_count;  // 3*7 = 21
        const int spatial = grid_h * grid_w;

        for (int a = 0; a < num_anchors; ++a) {
            const float anchor_w = MODEL_ANCHORS[branch_idx][a][0];
            const float anchor_h = MODEL_ANCHORS[branch_idx][a][1];

            for (int gy = 0; gy < grid_h; ++gy) {
                for (int gx = 0; gx < grid_w; ++gx) {
                    // 根据数据布局读取原始属性.
                    //   NCHW: data[(a*attr_count+k) * H*W + gy*W + gx]
                    //   NHWC: data[(gy*W+gx) * C + a*attr_count + k]
                    float val[32];  // 足够容纳 attr_count (本项目=7)
                    if (info.is_nchw) {
                        const int sp_off = gy * grid_w + gx;
                        for (int k = 0; k < attr_count; ++k) {
                            val[k] = info.data[(a * attr_count + k) * spatial + sp_off];
                        }
                    } else {
                        const int base = (gy * grid_w + gx) * channel + a * attr_count;
                        for (int k = 0; k < attr_count; ++k) {
                            val[k] = info.data[base + k];
                        }
                    }

                    // ---- 第一道过滤: logit 比较, 无 expf 开销 ----
                    // 当 obj_logit < obj_logit_thresh 时必然 sigmoid(obj_logit) < conf_threshold,
                    // 绝大多数背景格子在这里被跳过 (~98% 以上).
                    if (val[4] < obj_logit_thresh) {
                        continue;
                    }

                    // ---- 第二道过滤: 类别 logit 最大值 (同样无 expf) ----
                    int best_cls = 0;
                    float best_cls_logit = val[5];
                    for (int c = 1; c < num_classes; ++c) {
                        if (val[5 + c] > best_cls_logit) {
                            best_cls_logit = val[5 + c];
                            best_cls = c;
                        }
                    }
                    // 粗筛: sigmoid(obj) * sigmoid(cls) 的上界 < conf_threshold 则跳过.
                    // 当 max(obj_logit, cls_logit) 都大时才有可能通过, 用宽松条件:
                    if (best_cls_logit < cls_logit_thresh) {
                        continue;
                    }

                    // ---- 精确计算 (仅对极少数候选格子) ----
                    const float obj_conf     = sigmoid_f(val[4]);
                    const float best_cls_conf = sigmoid_f(best_cls_logit);
                    const float score = obj_conf * best_cls_conf;
                    if (score < conf_threshold) {
                        continue;
                    }

                    // YOLOv5 坐标解码公式:
                    //   cx = (sigmoid(tx)*2 - 0.5 + grid_x) * stride
                    //   cy = (sigmoid(ty)*2 - 0.5 + grid_y) * stride
                    //   w  = (sigmoid(tw)*2)^2 * anchor_w
                    //   h  = (sigmoid(th)*2)^2 * anchor_h
                    const float sx = sigmoid_f(val[0]);
                    const float sy = sigmoid_f(val[1]);
                    const float sw = sigmoid_f(val[2]) * 2.0f;
                    const float sh = sigmoid_f(val[3]) * 2.0f;

                    const float cx = (sx * 2.0f - 0.5f + (float)gx) * (float)stride;
                    const float cy = (sy * 2.0f - 0.5f + (float)gy) * (float)stride;
                    const float bw = sw * sw * anchor_w;
                    const float bh = sh * sh * anchor_h;

                    Detection det;
                    det.x1 = cx - bw * 0.5f;
                    det.y1 = cy - bh * 0.5f;
                    det.x2 = cx + bw * 0.5f;
                    det.y2 = cy + bh * 0.5f;
                    det.score = score;
                    det.class_id = best_cls;
                    if (is_valid_box(det)) {
                        detections.push_back(det);
                    }
                }
            }
        }
    }

    return detections;
}

// ========================================================================
//  dequantize_int8 — INT8 量化输出反量化
// ========================================================================
//
// RKNN 默认输出可能是 INT8 量化格式 (取决于量化配置).
// 如果你在 rknn_outputs_get 时使用 want_float=0 获取原始 INT8 以提升性能,
// 就需要先调用此函数把 INT8 数据还原为 float, 再传入 decode_yolov5_output.
//
// 公式: float_value = (int8_value - zero_point) * scale
//
// zero_point 和 scale 可从 rknn_tensor_attr 中获取:
//   attr.zp    → zero_point
//   attr.scale → scale
//
void dequantize_int8(
    const int8_t* int8_data,
    float* float_out,
    int count,
    int zp,
    float scale) {
#ifdef __ARM_NEON
    // NEON 路径: 每次处理 16 个 INT8，矢量化 int8 → float32 反量化
    // 吞吐量约为标量的 8～10x，176K 元素耗时 ~0.07ms vs 标量 ~0.7ms
    const float32x4_t vscale = vdupq_n_f32(scale);
    const float32x4_t vzp    = vdupq_n_f32(static_cast<float>(zp));
    int i = 0;
    for (; i <= count - 16; i += 16) {
        int8x16_t vs = vld1q_s8(int8_data + i);
        int16x8_t v16_lo = vmovl_s8(vget_low_s8(vs));
        int16x8_t v16_hi = vmovl_s8(vget_high_s8(vs));
        float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_lo)));
        float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_lo)));
        float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_hi)));
        float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_hi)));
        vst1q_f32(float_out + i,      vmulq_f32(vsubq_f32(f0, vzp), vscale));
        vst1q_f32(float_out + i + 4,  vmulq_f32(vsubq_f32(f1, vzp), vscale));
        vst1q_f32(float_out + i + 8,  vmulq_f32(vsubq_f32(f2, vzp), vscale));
        vst1q_f32(float_out + i + 12, vmulq_f32(vsubq_f32(f3, vzp), vscale));
    }
    // 剩余不足 16 个的尾部用标量处理
    for (; i < count; ++i) {
        float_out[i] = (static_cast<float>(int8_data[i]) - static_cast<float>(zp)) * scale;
    }
#else
    for (int i = 0; i < count; ++i) {
        float_out[i] = (static_cast<float>(int8_data[i]) - static_cast<float>(zp)) * scale;
    }
#endif
}
