#include "rknn_detector.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "file_utils.h"
#include "image_utils.h"

namespace {

// ----------------------------------------------------------------
//  调试辅助: 打印张量属性
// ----------------------------------------------------------------
// 在板端终端会输出类似:
//   index=0, name=output, n_dims=3, dims=[1, 25200, 7], ...
// 你可以据此确认 RKNN 模型的输入输出 shape 和量化参数.
void dump_tensor_attr(const rknn_tensor_attr* attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], "
           "n_elems=%d, size=%d, fmt=%d, type=%d, "
           "qnt_type=%d, zp=%d, scale=%f\n",
           attr->index,
           attr->name,
           attr->n_dims,
           attr->dims[0],
           attr->dims[1],
           attr->dims[2],
           attr->dims[3],
           attr->n_elems,
           attr->size,
           attr->fmt,
           attr->type,
           attr->qnt_type,
           attr->zp,
           attr->scale);
}

// ----------------------------------------------------------------
//  从输出张量属性推断 proposal 数 和 每行维度
// ----------------------------------------------------------------
//
// 对于本项目单输出 YOLOv5 模型, 输出张量形状为 [1, 25200, 7].
// 此函数的策略: 找张量各维度中最像 output_dim 的候选值
// (大于 5 且不超过 512), 然后用总元素数除以它得到 num_proposals.
//
// 本模型期望: output_dim=7, num_proposals=25200.
int infer_output_layout(const rknn_tensor_attr* attr, int* num_proposals, int* output_dim) {
    if (attr == NULL || num_proposals == NULL || output_dim == NULL) {
        return -1;
    }

    // 走过所有维度, 找最小的候选 output_dim.
    // 对 dims=[1, 25200, 7]: 候选值为 7.
    int candidate_dim = -1;
    for (int index = 0; index < attr->n_dims; ++index) {
        const int dim = attr->dims[index];
        if (dim > 5 && dim <= 512) {
            if (candidate_dim < 0 || dim < candidate_dim) {
                candidate_dim = dim;
            }
        }
    }

    if (candidate_dim < 0) {
        return -1;
    }

    // 总元素数 = n_elems = 1 * 25200 * 7 = 176400
    // num_proposals = 176400 / 7 = 25200
    if (attr->n_elems <= 0 || attr->n_elems % candidate_dim != 0) {
        return -1;
    }

    *output_dim = candidate_dim;
    *num_proposals = attr->n_elems / candidate_dim;
    return 0;
}

bool env_enabled(const char* name) {
    const char* value = getenv(name);
    if (value == NULL) return false;
    if (strcmp(value, "0") == 0 || strcmp(value, "false") == 0 || strcmp(value, "FALSE") == 0) {
        return false;
    }
    return true;
}

void print_debug_detections(const std::vector<Detection>& detections, const char* title, int max_count) {
    printf("[DEBUG] %s: %d\n", title, static_cast<int>(detections.size()));
    int n = static_cast<int>(detections.size());
    if (n > max_count) n = max_count;
    for (int i = 0; i < n; ++i) {
        const Detection& det = detections[i];
        printf("  #%d cls=%d score=%.4f box=(%.1f,%.1f,%.1f,%.1f)\n",
               i, det.class_id, det.score, det.x1, det.y1, det.x2, det.y2);
    }
}

}  // namespace

// ========================================================================
//  ensure_rgb888 — 红外灰度图 → 3 通道转换
// ========================================================================
//
// 红外摄像头或部分图像格式可能输出单通道灰度 (IMAGE_FORMAT_GRAY8).
// 但本模型训练时的输入是 3 通道 (OpenCV 把灰度 IR 图自动读为 BGR),
// 所以推理时必须保证输入也是 3 通道.
//
// 处理策略:
//   - 已经是 RGB888: 直接返回原始图像指针, 无需额外内存
//   - 是 GRAY8: 分配新的 RGB888 缓冲区, 把灰度值复制到 R/G/B 三通道
//   - 其他格式: 打印错误, 返回 NULL
//
image_buffer_t* ensure_rgb888(
    image_buffer_t* src_image,
    image_buffer_t* rgb_buffer,
    bool* need_free) {
    if (src_image == NULL || rgb_buffer == NULL || need_free == NULL) {
        return NULL;
    }

    *need_free = false;

    // 情况 1: 已经是 RGB888, 直接返回
    if (src_image->format == IMAGE_FORMAT_RGB888) {
        return src_image;
    }

    // 情况 2: 单通道灰度红外图, 需要复制为 3 通道
    if (src_image->format == IMAGE_FORMAT_GRAY8) {
        memset(rgb_buffer, 0, sizeof(*rgb_buffer));
        rgb_buffer->width = src_image->width;
        rgb_buffer->height = src_image->height;
        rgb_buffer->format = IMAGE_FORMAT_RGB888;
        const int pixel_count = src_image->width * src_image->height;
        rgb_buffer->size = pixel_count * 3;
        rgb_buffer->virt_addr = (unsigned char*)malloc(rgb_buffer->size);
        if (rgb_buffer->virt_addr == NULL) {
            printf("[ERROR] ensure_rgb888: malloc failed (%d bytes)\n", rgb_buffer->size);
            return NULL;
        }

        // 灰度 → RGB: 每个像素的灰度值复制到 R, G, B 三个通道.
        // 这和 OpenCV 训练时 cv2.imread 读灰度 JPEG 的行为一致.
        const unsigned char* gray = src_image->virt_addr;
        unsigned char* rgb = rgb_buffer->virt_addr;
        for (int i = 0; i < pixel_count; ++i) {
            rgb[i * 3 + 0] = gray[i];  // R = 灰度
            rgb[i * 3 + 1] = gray[i];  // G = 灰度
            rgb[i * 3 + 2] = gray[i];  // B = 灰度
        }

        *need_free = true;
        printf("[INFO] 灰度红外图 (%dx%d) 已转换为 3 通道 RGB888\n",
               src_image->width, src_image->height);
        return rgb_buffer;
    }

    // 情况 3: 不支持的格式
    printf("[ERROR] ensure_rgb888: unsupported format %d, only RGB888 and GRAY8\n",
           src_image->format);
    return NULL;
}

// ========================================================================
//  init_rknn_model — 加载并校验 RKNN 模型
// ========================================================================
int init_rknn_model(const char* model_path, RknnAppContext* app_ctx) {
    if (model_path == NULL || app_ctx == NULL) {
        return -1;
    }

    // 先清零上下文, 避免未初始化字段.
    memset(app_ctx, 0, sizeof(*app_ctx));

    // ---------- 1. 读取 .rknn 文件到内存 ----------
    char* model_data = NULL;
    const int model_size = read_data_from_file(model_path, &model_data);
    if (model_size <= 0 || model_data == NULL) {
        printf("[ERROR] load model failed: %s\n", model_path);
        return -1;
    }

    // ---------- 2. 创建 RKNN 运行时会话 ----------
    int ret = rknn_init(&app_ctx->rknn_ctx, model_data, model_size, 0, NULL);
    free(model_data);   // 初始化后原始数据即可释放
    if (ret != RKNN_SUCC) {
        printf("[ERROR] rknn_init failed: %d\n", ret);
        return -1;
    }

    // ---------- 3. 查询输入输出张量数量 ----------
    ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(app_ctx->io_num));
    if (ret != RKNN_SUCC) {
        printf("[ERROR] rknn_query in/out num failed: %d\n", ret);
        return -1;
    }

    // 本部署代码只支持 "1 个输入 + 1 个输出" 的单输出 YOLOv5 模型.
    // 本部署代码支持 1 个输出 (单输出已解码) 或 3 个输出 (raw conv).
    if (app_ctx->io_num.n_input != 1) {
        printf("[ERROR] unsupported input count: %u (expect 1)\n", app_ctx->io_num.n_input);
        return -1;
    }
    if (app_ctx->io_num.n_output != 1 && app_ctx->io_num.n_output != 3) {
        printf("[ERROR] unsupported output count: %u (expect 1 or 3)\n", app_ctx->io_num.n_output);
        return -1;
    }

    // ---------- 4. 分配并查询输入输出张量属性 ----------
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * app_ctx->io_num.n_input);
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * app_ctx->io_num.n_output);
    if (app_ctx->input_attrs == NULL || app_ctx->output_attrs == NULL) {
        printf("[ERROR] malloc tensor attrs failed\n");
        return -1;
    }

    memset(app_ctx->input_attrs, 0, sizeof(rknn_tensor_attr) * app_ctx->io_num.n_input);
    memset(app_ctx->output_attrs, 0, sizeof(rknn_tensor_attr) * app_ctx->io_num.n_output);

    printf("--- input tensors ---\n");
    for (unsigned int index = 0; index < app_ctx->io_num.n_input; ++index) {
        app_ctx->input_attrs[index].index = index;
        ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_INPUT_ATTR, &app_ctx->input_attrs[index], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("[ERROR] query input attr[%u] failed: %d\n", index, ret);
            return -1;
        }
        dump_tensor_attr(&app_ctx->input_attrs[index]);
    }

    printf("--- output tensors ---\n");
    for (unsigned int index = 0; index < app_ctx->io_num.n_output; ++index) {
        app_ctx->output_attrs[index].index = index;
        ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &app_ctx->output_attrs[index], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("[ERROR] query output attr[%u] failed: %d\n", index, ret);
            return -1;
        }
        dump_tensor_attr(&app_ctx->output_attrs[index]);
    }

    // ---------- 5. 解析输入尺寸 ----------
    // RKNN 模型可能是 NCHW 或 NHWC 布局, 分别处理.
    if (app_ctx->input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        // dims: [batch, channel, height, width]
        app_ctx->model_channel = app_ctx->input_attrs[0].dims[1];
        app_ctx->model_height = app_ctx->input_attrs[0].dims[2];
        app_ctx->model_width = app_ctx->input_attrs[0].dims[3];
    } else {
        // NHWC: dims: [batch, height, width, channel]
        app_ctx->model_height = app_ctx->input_attrs[0].dims[1];
        app_ctx->model_width = app_ctx->input_attrs[0].dims[2];
        app_ctx->model_channel = app_ctx->input_attrs[0].dims[3];
    }

    // ---------- 6. 推断输出布局 ----------
    if (app_ctx->io_num.n_output == 1) {
        // ---- 单输出模式: [1, 25200, 7], Detect 头已解码 ----
        app_ctx->is_3branch = false;
        ret = infer_output_layout(&app_ctx->output_attrs[0],
                                  &app_ctx->num_proposals, &app_ctx->output_dim);
        if (ret != 0 || app_ctx->output_dim <= 5) {
            printf("[ERROR] failed to infer output layout. "
                   "This model may not be standard single-output YOLOv5.\n");
            return -1;
        }
        app_ctx->num_classes = app_ctx->output_dim - 5;
    } else {
        // ---- 3-branch 模式: 3 个检测头的 raw conv 输出 ----
        app_ctx->is_3branch = true;

        // 按 n_elems 降序排列, 确定 P3(最大)/P4/P5(最小) 与 rknn 输出索引的映射.
        app_ctx->branch_index[0] = 0;
        app_ctx->branch_index[1] = 1;
        app_ctx->branch_index[2] = 2;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2 - i; ++j) {
                if (app_ctx->output_attrs[app_ctx->branch_index[j]].n_elems <
                    app_ctx->output_attrs[app_ctx->branch_index[j + 1]].n_elems) {
                    int tmp = app_ctx->branch_index[j];
                    app_ctx->branch_index[j] = app_ctx->branch_index[j + 1];
                    app_ctx->branch_index[j + 1] = tmp;
                }
            }
        }

        // 从最大输出 (P3, stride=8, grid=80x80) 推断 output_dim.
        // n_elems = grid_h * grid_w * num_anchors * output_dim
        const rknn_tensor_attr* p3 = &app_ctx->output_attrs[app_ctx->branch_index[0]];
        int p3_spatial = MODEL_GRID_SIZES[0] * MODEL_GRID_SIZES[0] * MODEL_NUM_ANCHORS;
        if (p3->n_elems <= 0 || p3->n_elems % p3_spatial != 0) {
            printf("[ERROR] P3 output n_elems=%d 不能被 %d 整除\n",
                   p3->n_elems, p3_spatial);
            return -1;
        }
        app_ctx->output_dim = p3->n_elems / p3_spatial;
        if (app_ctx->output_dim <= 5) {
            printf("[ERROR] 3-branch 推断 output_dim=%d, 不合理\n", app_ctx->output_dim);
            return -1;
        }
        app_ctx->num_classes = app_ctx->output_dim - 5;
        app_ctx->num_proposals = MODEL_NUM_PROPOSALS;

        // 验证 P4, P5 的 n_elems 一致性.
        for (int b = 1; b < 3; ++b) {
            const rknn_tensor_attr* attr = &app_ctx->output_attrs[app_ctx->branch_index[b]];
            int expected = MODEL_GRID_SIZES[b] * MODEL_GRID_SIZES[b]
                         * MODEL_NUM_ANCHORS * app_ctx->output_dim;
            if (attr->n_elems != expected) {
                printf("[WARN] branch P%d (stride %d): n_elems=%d, 期望 %d\n",
                       b + 3, MODEL_STRIDES[b], attr->n_elems, expected);
            }
        }
    }

    // ---------- 7. 模型信息与一致性校验 ----------
    printf("\n=== 模型信息 ===\n");
    printf("  input:    %d x %d x %d\n",
           app_ctx->model_width, app_ctx->model_height, app_ctx->model_channel);
    if (app_ctx->is_3branch) {
        printf("  output:   3-branch (raw conv, 需完整解码)\n");
        for (int b = 0; b < 3; ++b) {
            printf("    P%d/stride=%d: grid %dx%d, n_elems=%d\n",
                   b + 3, MODEL_STRIDES[b],
                   MODEL_GRID_SIZES[b], MODEL_GRID_SIZES[b],
                   app_ctx->output_attrs[app_ctx->branch_index[b]].n_elems);
        }
    } else {
        printf("  output:   %d proposals x %d dim (单输出已解码)\n",
               app_ctx->num_proposals, app_ctx->output_dim);
    }
    printf("  classes:  %d\n", app_ctx->num_classes);

    if (app_ctx->model_width != MODEL_INPUT_SIZE || app_ctx->model_height != MODEL_INPUT_SIZE) {
        printf("[WARN] 输入尺寸 %dx%d != 期望 %dx%d\n",
               app_ctx->model_width, app_ctx->model_height,
               MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    }
    if (app_ctx->num_classes != MODEL_NUM_CLASSES) {
        printf("[WARN] 类别数 %d != 期望 %d\n",
               app_ctx->num_classes, MODEL_NUM_CLASSES);
    }
    if (!app_ctx->is_3branch) {
        if (app_ctx->num_proposals != MODEL_NUM_PROPOSALS) {
            printf("[WARN] proposal 数 %d != 期望 %d\n",
                   app_ctx->num_proposals, MODEL_NUM_PROPOSALS);
        }
        if (app_ctx->output_dim != MODEL_OUTPUT_DIM) {
            printf("[WARN] 输出维度 %d != 期望 %d\n",
                   app_ctx->output_dim, MODEL_OUTPUT_DIM);
        }
    }

    printf("=== 模型加载完成 ===\n\n");
    return 0;
}

// ========================================================================
//  release_rknn_model — 释放资源
// ========================================================================
int release_rknn_model(RknnAppContext* app_ctx) {
    if (app_ctx == NULL) {
        return -1;
    }
    // 释放我们 malloc 的属性数组.
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    // 销毁 RKNN 会话.
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

// ========================================================================
//  inference_rknn_model — 红外图像完整推理流水线
// ========================================================================
//
// 整体流程:
//   红外原图 (src_image, 可能是灰度或 3 通道)
//     ↓ ensure_rgb888: 若为单通道灰度则复制为 3 通道
//     ↓ letterbox 缩放到 640×640, 灰色(114)填充
//   预处理后图 (dst_image, RGB888 uint8)
//     ↓ rknn_inputs_set
//   NPU 推理
//     ↓ rknn_run
//   获取输出 → [1, 25200, 7] float
//     ↓ decode_yolov5_output:  解码 + 阈值过滤
//     ↓ scale_detections:     letterbox → 原图坐标
//     ↓ apply_nms:            按类别 NMS 去重
//   输出 detections
//
int inference_rknn_model(
    RknnAppContext* app_ctx,
    image_buffer_t* src_image,
    float conf_threshold,
    float nms_threshold,
    std::vector<Detection>* detections) {
    if (app_ctx == NULL || src_image == NULL || detections == NULL) {
        return -1;
    }

    if (conf_threshold < 0.01f) conf_threshold = 0.01f;
    if (conf_threshold > 0.95f) conf_threshold = 0.95f;
    if (nms_threshold < 0.05f) nms_threshold = 0.05f;
    if (nms_threshold > 0.95f) nms_threshold = 0.95f;

    const bool debug_box = env_enabled("RKNN_DEBUG_BOX");

    detections->clear();

    // -------- 红外图像通道转换 --------
    // 红外摄像头可能输出单通道灰度 (IMAGE_FORMAT_GRAY8),
    // 但模型训练时 OpenCV 把灰度 IR 图自动读为 3 通道 BGR,
    // 所以推理时也必须保证输入是 3 通道.
    //
    // ensure_rgb888 会自动处理:
    //   - JPEG/PNG 文件: read_image 已读为 RGB888, 直接透传
    //   - 摄像头灰度帧: 自动复制灰度值到 R/G/B 三通道
    image_buffer_t rgb_buffer;
    bool need_free_rgb = false;
    image_buffer_t* rgb_image = ensure_rgb888(src_image, &rgb_buffer, &need_free_rgb);
    if (rgb_image == NULL) {
        printf("[ERROR] ensure_rgb888 failed\n");
        return -1;
    }

    // -------- 预处理: letterbox 缩放 + 灰边填充 --------
    // 分配和模型输入一样大小的 3 通道缓冲区.
    image_buffer_t dst_image;
    memset(&dst_image, 0, sizeof(dst_image));
    dst_image.width = app_ctx->model_width;    // 640
    dst_image.height = app_ctx->model_height;  // 640
    dst_image.format = IMAGE_FORMAT_RGB888;
    dst_image.size = get_image_size(&dst_image);
    dst_image.virt_addr = (unsigned char*)malloc(dst_image.size);
    if (dst_image.virt_addr == NULL) {
        printf("[ERROR] malloc input buffer failed\n");
        if (need_free_rgb) free(rgb_buffer.virt_addr);
        return -1;
    }

    // letterbox 保持宽高比不变, 用灰色(114)填充多余区域.
    // 填充值 114 是 YOLOv5 训练时的默认值, 对红外灰度图像同样适用.
    letterbox_t letterbox;
    memset(&letterbox, 0, sizeof(letterbox));
    int ret = convert_image_with_letterbox(rgb_image, &dst_image, &letterbox, 114);

    // rgb_buffer 用完即可释放, 后续只用 dst_image
    if (need_free_rgb) {
        free(rgb_buffer.virt_addr);
        need_free_rgb = false;
    }
    if (ret < 0) {
        printf("[ERROR] convert_image_with_letterbox failed: %d\n", ret);
        free(dst_image.virt_addr);
        return -1;
    }

    // -------- 设置输入 --------
    // 输入格式: uint8 NHWC RGB888, 尺寸 640×640×3.
    // 虽然红外图像三通道的像素值相同, 但 RKNN 运行时在输入端做的
    // mean/std 归一化 (mean=[0,0,0], std=[255,255,255]) 会正确处理.
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    input.buf = dst_image.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, &input);
    if (ret != RKNN_SUCC) {
        printf("[ERROR] rknn_inputs_set failed: %d\n", ret);
        free(dst_image.virt_addr);
        return -1;
    }

    // -------- NPU 推理 --------
    ret = rknn_run(app_ctx->rknn_ctx, NULL);
    if (ret != RKNN_SUCC) {
        printf("[ERROR] rknn_run failed: %d\n", ret);
        free(dst_image.virt_addr);
        return -1;
    }

    // -------- 获取输出 & 后处理 --------
    std::vector<Detection> decoded;

    if (app_ctx->is_3branch) {
        // ---- 3-branch: 获取 3 个输出, 完整解码 ----
        rknn_output outputs[3];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < 3; ++i) {
            outputs[i].index = i;
            outputs[i].want_float = 1;
        }
        ret = rknn_outputs_get(app_ctx->rknn_ctx, 3, outputs, NULL);
        if (ret != RKNN_SUCC) {
            printf("[ERROR] rknn_outputs_get (3-branch) failed: %d\n", ret);
            free(dst_image.virt_addr);
            return -1;
        }

        BranchInfo branches[3];
        for (int b = 0; b < 3; ++b) {
            int idx = app_ctx->branch_index[b];
            const rknn_tensor_attr* attr = &app_ctx->output_attrs[idx];
            branches[b].data = (const float*)outputs[idx].buf;
            branches[b].grid_h = MODEL_GRID_SIZES[b];
            branches[b].grid_w = MODEL_GRID_SIZES[b];
            branches[b].stride = MODEL_STRIDES[b];
            branches[b].is_nchw = (attr->fmt == RKNN_TENSOR_NCHW);
        }

        decoded = decode_yolov5_3branch_output(
            branches, MODEL_NUM_ANCHORS, app_ctx->num_classes, conf_threshold);

        rknn_outputs_release(app_ctx->rknn_ctx, 3, outputs);
    } else {
        // ---- 单输出: [1, 25200, 7] 已解码 ----
        rknn_output output;
        memset(&output, 0, sizeof(output));
        output.index = 0;
        output.want_float = 1;

        ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, &output, NULL);
        if (ret != RKNN_SUCC) {
            printf("[ERROR] rknn_outputs_get failed: %d\n", ret);
            free(dst_image.virt_addr);
            return -1;
        }

        const float* raw_output = (const float*)output.buf;
        decoded = decode_yolov5_output(
            raw_output, app_ctx->num_proposals, app_ctx->output_dim, conf_threshold);

        rknn_outputs_release(app_ctx->rknn_ctx, 1, &output);
    }

    // 坐标映射 + NMS (两种输出格式共用)
    const int decoded_count = static_cast<int>(decoded.size());
    scale_detections(decoded, letterbox, src_image->width, src_image->height);
    *detections = apply_nms(decoded, nms_threshold);

    if (debug_box) {
        static int debug_frames = 0;
        if (debug_frames < 20) {
            printf("[DEBUG] frame=%d conf=%.2f nms=%.2f letterbox(scale=%.6f,x_pad=%d,y_pad=%d) decoded=%d nms_out=%d mode=%s\n",
                   debug_frames,
                   conf_threshold,
                   nms_threshold,
                   letterbox.scale,
                   letterbox.x_pad,
                   letterbox.y_pad,
                   decoded_count,
                   static_cast<int>(detections->size()),
                     "float_decode");
            print_debug_detections(*detections, "post_nms_top", 8);
        }
        ++debug_frames;
    }

    // -------- 清理 --------
    free(dst_image.virt_addr);
    return 0;
}