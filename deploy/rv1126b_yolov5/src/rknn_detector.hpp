#pragma once

#include <vector>

#include "rknn_api.h"
#include "common.h"
#include "postprocess.hpp"

// ========================================================================
//  RknnAppContext — RKNN 推理上下文 (红外目标检测)
// ========================================================================
//
// 集中保存:
//   1. RKNN 运行时句柄 (rknn_ctx)
//   2. 输入输出张量属性 (input_attrs / output_attrs)
//   3. 模型输入尺寸 (model_width / model_height / model_channel)
//   4. 推断的输出参数 (output_dim / num_proposals / num_classes)
//
// ※ 关于红外图像的通道数:
//   虽然红外图像本质上是单通道灰度, 但训练时 OpenCV 自动读为 3 通道,
//   所以模型输入仍为 3 通道 (三个通道的值完全相同).
//   如果板端红外摄像头输出单通道灰度帧, 推理前会自动复制为 3 通道.
//
// 本项目模型的期望值:
//   model_width  = 640   model_height  = 640   model_channel = 3
//   output_dim   = 7     num_proposals = 25200  num_classes   = 2
// 支持两种输出格式:
//   单输出 (export=True):  [1, 25200, 7] — 已解码, 只需阈值过滤 + NMS
//   3-branch (raw conv): 3 个输出, 各为 P3/P4/P5 原始数据, 需完整解码
struct RknnAppContext {
    rknn_context rknn_ctx;              // RKNN 会话句柄
    rknn_input_output_num io_num;       // 输入/输出张量数量
    rknn_tensor_attr* input_attrs;      // 输入张量属性数组
    rknn_tensor_attr* output_attrs;     // 输出张量属性数组
    int model_channel;                  // 输入通道数 (3, 即使是红外灰度也用 3 通道)
    int model_width;                    // 输入宽 (应为 640)
    int model_height;                   // 输入高 (应为 640)
    int output_dim;                     // 输出每行维度 (应为 7)
    int num_proposals;                  // 输出 proposal 总数 (单输出=25200)
    int num_classes;                    // 类别数 (应为 2)
    bool is_3branch;                    // true = 3 输出 (raw conv), false = 1 输出 (已解码)
    int branch_index[3];                // 3-branch 时: branches[i] 对应的 rknn 输出索引
    unsigned char* input_buffer;        // 模型输入数据指针，指向 input_mem->virt_addr（零拷贝）或 malloc 缓冲
    int input_buffer_size;              // 输入缓冲区大小（字节）
    rknn_tensor_mem* input_mem;         // 零拷贝输入共享内存（rknn_create_mem，NPU 直接访问）
    float*  output_float_bufs[3];       // 3-branch float32 输出缓冲（NEON dequant 目标）
    int     output_float_buf_sizes[3];  // 各 float 缓冲字节数
    int8_t* output_int8_bufs[3];        // 3-branch INT8 输出缓冲（rknn_outputs_get want_float=0 prealloc）
    int     output_int8_buf_sizes[3];   // 各 INT8 缓冲字节数
    double last_npu_ms;                 // 最近一帧 rknn_run 纯 NPU 时延（毫秒）
};

// 初始化 RKNN 模型:
//   1. 读取 .rknn 文件
//   2. rknn_init 创建推理会话
//   3. 查询输入输出张量属性
//   4. 推断输出布局 (num_proposals / output_dim / num_classes)
//   5. 与 MODEL_* 常量做一致性校验
int init_rknn_model(const char* model_path, RknnAppContext* app_ctx);

// 释放模型和相关内存.
int release_rknn_model(RknnAppContext* app_ctx);

// 确保图像是 3 通道 RGB888 格式.
//
// 红外摄像头或特殊格式可能输出单通道灰度 (IMAGE_FORMAT_GRAY8).
// 此函数自动检测:
//   - 如果已经是 RGB888: 直接返回原图指针, *need_free = false
//   - 如果是 GRAY8: 分配新 3 通道缓冲区, 把灰度复制到 R/G/B,
//     返回新缓冲区指针, *need_free = true (调用者负责释放)
//
// 返回 NULL 表示失败.
image_buffer_t* ensure_rgb888(
    image_buffer_t* src_image,
    image_buffer_t* rgb_buffer,
    bool* need_free);

// 执行一次完整推理 (预处理 → NPU 推理 → 后处理):
//
//   红外图像 (可能为灰度或 3 通道)
//     ↓ ensure_rgb888: 确保 3 通道
//     ↓ letterbox: 缩放到 640×640, 灰色(114)填充
//     ↓ rknn_inputs_set
//     ↓ rknn_run (NPU 推理)
//     ↓ rknn_outputs_get → [1, 25200, 7] float
//     ↓ decode → scale → NMS
//   输出 detections
int inference_rknn_model(
    RknnAppContext* app_ctx,
    image_buffer_t* src_image,
    float conf_threshold,
    float nms_threshold,
    std::vector<Detection>* detections);

// 执行推理 (跳过预处理, 使用已填充的 input_buffer):
//
//   调用者负责在调用前将 640×640×3 RGB888 数据写入 app_ctx->input_buffer.
//   letterbox 和 src 宽高由调用者提供, 用于坐标反映射.
//   可节省约 7~9ms 的 CPU letterbox 开销.
int inference_rknn_model_preloaded(
    RknnAppContext* app_ctx,
    const letterbox_t* letterbox,
    int src_width,
    int src_height,
    float conf_threshold,
    float nms_threshold,
    std::vector<Detection>* detections);
