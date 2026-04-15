#pragma once

#include <cstdint>
#include <vector>

#include "image_utils.h"

// ========================================================================
//  本项目 YOLOv5s 红外目标检测模型 —— 固定参数
// ========================================================================
//
//  ※ 这是一个红外 (热成像 / IR) 检测项目，不是普通 RGB 彩色检测。
//
//  训练时 FLIR 红外图像被 OpenCV 自动读取为 3 通道 (BGR, 三通道值相同)，
//  因此模型输入仍然是 3 通道，但每个通道的像素值实际上都是灰度亮度。
//
//  输入分辨率:   640 × 640 × 3 (红外灰度图复制为三通道)
//  类别数 nc:    2 (person=0, car=1)
//  检测头层数:   3 (P3 / P4 / P5)
//  每层锚框数:   3
//
//  三个检测头的网格大小 & 对应 stride:
//    P3  stride= 8  → 640/8  = 80  → 80×80×3 = 19200 proposals
//    P4  stride=16  → 640/16 = 40  → 40×40×3 =  4800 proposals
//    P5  stride=32  → 640/32 = 20  → 20×20×3 =  1200 proposals
//    合计: 19200 + 4800 + 1200 = 25200 proposals
//
//  训练 yaml 中定义的锚框 (像素):
//    P3/8:   (10,13)   (16,30)   (33,23)
//    P4/16:  (30,61)   (62,45)   (59,119)
//    P5/32:  (116,90)  (156,198) (373,326)
//
//  单个 proposal 的维度: 5 + nc = 7
//    [cx, cy, w, h, obj_conf, person_conf, car_conf]
//
//  ONNX 导出 (export=True) 时 Detect 头已完成:
//    1) sigmoid 激活
//    2) grid 偏移解码: xy = (sigmoid(xy)*2 - 0.5 + grid) * stride
//    3) anchor 缩放:   wh = (sigmoid(wh)*2)^2 * anchor
//  因此 RKNN 输出张量 [1, 25200, 7] 里的 cx/cy/w/h 已是
//  640×640 像素空间下的绝对坐标, obj_conf/cls_conf 已经 sigmoid。
//  → 后处理 **不需要** 再做 grid 解码或 sigmoid, 直接阈值过滤即可.
// ========================================================================

static const int MODEL_INPUT_SIZE    = 640;   // 输入图像边长
static const int MODEL_NUM_CLASSES   = 2;     // person=0, car=1
static const int MODEL_OUTPUT_DIM    = 7;     // 5 + MODEL_NUM_CLASSES
static const int MODEL_NUM_PROPOSALS = 25200; // (80*80+40*40+20*20)*3

// 三个检测头的网格边长, 按顺序排列.
static const int MODEL_GRID_SIZES[]  = {80, 40, 20};
// 对应的 stride.
static const int MODEL_STRIDES[]     = {8, 16, 32};
// 每层的锚框数.
static const int MODEL_NUM_ANCHORS   = 3;

// 锚框像素值 (3 个检测头 × 3 个锚框 × [宽, 高]).
// 这些值来自训练 yaml, 在 3-branch 解码时使用.
static const float MODEL_ANCHORS[3][3][2] = {
    {{10.f, 13.f}, {16.f, 30.f}, {33.f, 23.f}},      // P3/stride=8
    {{30.f, 61.f}, {62.f, 45.f}, {59.f, 119.f}},      // P4/stride=16
    {{116.f, 90.f}, {156.f, 198.f}, {373.f, 326.f}},   // P5/stride=32
};

// 3-branch 解码时每个分支的描述信息.
struct BranchInfo {
    const float* data;   // 浮点数据缓冲区
    int grid_h;          // 网格高度 (如 80)
    int grid_w;          // 网格宽度 (如 80)
    int stride;          // 步长 (8/16/32)
    bool is_nchw;        // 数据布局: true=[1,C,H,W], false=[1,H,W,C]
};

// ========================================================================
//  检测结果结构
// ========================================================================

// 单个检测结果。
// 统一用左上角 (x1,y1) 和右下角 (x2,y2) 表示边界框。
struct Detection {
    float x1;       // 左上角 x
    float y1;       // 左上角 y
    float x2;       // 右下角 x
    float y2;       // 右下角 y
    float score;    // 最终得分 = obj_conf * cls_conf
    int class_id;   // 类别编号: 0=person, 1=car
};

// ========================================================================
//  后处理函数
// ========================================================================

// 解码单输出 YOLOv5 的 RKNN 推理结果.
//
// raw_output  — RKNN 输出缓冲区, 浮点, 长度 = num_proposals * output_dim
// num_proposals — 本模型为 25200
// output_dim    — 本模型为 7 (即 [cx, cy, w, h, obj, person, car])
// conf_threshold — 联合置信度阈值, 低于此值的 proposal 直接丢弃
//
// 对于本模型, raw_output 布局为:
//   row[0]  = cx   (已解码, 640 像素空间)
//   row[1]  = cy
//   row[2]  = w
//   row[3]  = h
//   row[4]  = obj_conf  (已 sigmoid)
//   row[5]  = person_conf (已 sigmoid)
//   row[6]  = car_conf    (已 sigmoid)
std::vector<Detection> decode_yolov5_output(
    const float* raw_output,
    int num_proposals,
    int output_dim,
    float conf_threshold);

// 把 letterbox 空间中的检测框映射回原图坐标.
//
// letterbox —— convert_image_with_letterbox 产生的缩放与填充信息
// 映射公式:
//   orig_x = (letterbox_x - pad_x) / scale
void scale_detections(
    std::vector<Detection>& detections,
    const letterbox_t& letterbox,
    int original_width,
    int original_height);

// 按类别做非极大值抑制 (NMS).
// 同一类别内, 高分框优先保留, 与其 IoU >= iou_threshold 的低分框被删除.
std::vector<Detection> apply_nms(
    const std::vector<Detection>& detections,
    float iou_threshold);

// 解码 3-branch YOLOv5 输出 (RKNN 模型有 3 个输出张量时使用).
//
// 3-branch 输出是三个检测头的原始 conv 输出 (未经 sigmoid/grid 解码),
// 需要完整的解码流程: sigmoid → grid 偏移 → anchor 缩放.
//
// branches[3] — P3/P4/P5 三个分支的信息 (按 stride 8/16/32 顺序)
// num_anchors — 每个网格点的锚框数 (3)
// num_classes — 类别数 (2)
std::vector<Detection> decode_yolov5_3branch_output(
    const BranchInfo branches[3],
    int num_anchors,
    int num_classes,
    float conf_threshold);

// ========================================================================
//  INT8 量化输出反量化 (可选)
// ========================================================================
//
// 如果你在 rknn_outputs_get 时使用 want_float=0 获取原始 INT8 输出,
// 需要先调用此函数把 INT8 数据反量化为 float, 再传入 decode_yolov5_output.
//
// 反量化公式:  float_value = (int8_value - zero_point) * scale
//
// 参数:
//   int8_data  — 原始 INT8 缓冲区
//   float_out  — 输出 float 缓冲区 (调用者预分配, 大小 = count)
//   count      — 元素总数 = num_proposals * output_dim = 25200 * 7 = 176400
//   zp         — 零点, 来自 rknn_tensor_attr.zp
//   scale      — 缩放因子, 来自 rknn_tensor_attr.scale
void dequantize_int8(
    const int8_t* int8_data,
    float* float_out,
    int count,
    int zp,
    float scale);