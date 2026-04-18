#include "rknn_detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

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

double now_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int convert_image_with_exact_letterbox(
    image_buffer_t* src_image,
    image_buffer_t* dst_image,
    letterbox_t* letterbox,
    char color) {
    if (src_image == NULL || dst_image == NULL) {
        return -1;
    }
    if (src_image->width <= 0 || src_image->height <= 0 ||
        dst_image->width <= 0 || dst_image->height <= 0) {
        return -1;
    }

    const int src_w = src_image->width;
    const int src_h = src_image->height;
    const int dst_w = dst_image->width;
    const int dst_h = dst_image->height;

    const float scale = std::min(static_cast<float>(dst_w) / static_cast<float>(src_w),
                                 static_cast<float>(dst_h) / static_cast<float>(src_h));

    const int resize_w = std::max(1, static_cast<int>(roundf(static_cast<float>(src_w) * scale)));
    const int resize_h = std::max(1, static_cast<int>(roundf(static_cast<float>(src_h) * scale)));

    const float pad_w = (static_cast<float>(dst_w) - static_cast<float>(resize_w)) * 0.5f;
    const float pad_h = (static_cast<float>(dst_h) - static_cast<float>(resize_h)) * 0.5f;

    int left = static_cast<int>(roundf(pad_w - 0.1f));
    int top = static_cast<int>(roundf(pad_h - 0.1f));
    left = std::max(0, left);
    top = std::max(0, top);

    if (left + resize_w > dst_w) left = std::max(0, dst_w - resize_w);
    if (top + resize_h > dst_h) top = std::max(0, dst_h - resize_h);

    image_rect_t src_box;
    src_box.left = 0;
    src_box.top = 0;
    src_box.right = src_w - 1;
    src_box.bottom = src_h - 1;

    image_rect_t dst_box;
    dst_box.left = left;
    dst_box.top = top;
    dst_box.right = left + resize_w - 1;
    dst_box.bottom = top + resize_h - 1;

    if (letterbox != NULL) {
        letterbox->scale = scale;
        letterbox->x_pad = left;
        letterbox->y_pad = top;
    }

    return convert_image(src_image, dst_image, &src_box, &dst_box, color);
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
    // RKNN_FLAG_ENABLE_SRAM: 将部分中间激活/权重放入片上 SRAM (带宽远高于 DRAM),
    // 对 RV1126B 单 NPU 核心可减少 DRAM 访问，降低 NPU 执行延迟 2~4ms.
    //
    // RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE: 跳过 rknn_run 前内部的 input cache flush.
    // 安全前提: 输入内存必须是非缓存 (NON_CACHEABLE) 或 CPU 手动 flush 后才能用此标志.
    // 我们将展开 rknn_create_mem2 + RKNN_FLAG_MEMORY_NON_CACHEABLE 创建非缓存输入内存,
    // 因此这里可以安全加入此标志，节省每帧约 1ms.
    const uint32_t rknn_flags = RKNN_FLAG_ENABLE_SRAM
                              | RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE;
    int ret = rknn_init(&app_ctx->rknn_ctx, model_data, model_size, rknn_flags, NULL);
    free(model_data);   // 初始化后原始数据即可释放
    if (ret != RKNN_SUCC) {
        printf("[ERROR] rknn_init failed: %d\n", ret);
        return -1;
    }

    // 尝试绑定所有可用 NPU 核心, 在多核平台 (如 RK3588) 可并行降低延迟;
    // 在 RV1126B 单核平台此调用会静默使用 CORE_0, 无副作用.
    ret = rknn_set_core_mask(app_ctx->rknn_ctx, RKNN_NPU_CORE_ALL);
    if (ret != RKNN_SUCC) {
        printf("[WARN] rknn_set_core_mask failed (non-fatal): %d\n", ret);
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
    app_ctx->last_npu_ms = 0.0;
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

    // ---------- 8. 零拷贝输入: rknn_create_mem2 (NON_CACHEABLE) + rknn_set_io_mem ----------
    // 非缓存内存 (NON_CACHEABLE) 的优势:
    //   1. CPU 写入直接达到 DRAM, NPU 读取时无需 cache 同步操作 (~1ms cache flush)
    //   2. 配合 RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE, rknn_run 前完全跳过 flush 步骤
    // 代价: CPU 写入速度比缓存内存稍慢 (letterbox 内核已用 OpenCV NEON 优化, 影响可忽略)
    app_ctx->input_mem = NULL;
    memset(app_ctx->output_float_bufs,      0, sizeof(app_ctx->output_float_bufs));
    memset(app_ctx->output_float_buf_sizes,  0, sizeof(app_ctx->output_float_buf_sizes));
    memset(app_ctx->output_int8_bufs,        0, sizeof(app_ctx->output_int8_bufs));
    memset(app_ctx->output_int8_buf_sizes,   0, sizeof(app_ctx->output_int8_buf_sizes));

    // 配置输入 attr: UINT8 NHWC (NPU 内部做 scale/zp 归一化)
    app_ctx->input_attrs[0].type = RKNN_TENSOR_UINT8;
    app_ctx->input_attrs[0].fmt  = RKNN_TENSOR_NHWC;
    uint32_t input_size = app_ctx->input_attrs[0].size_with_stride;
    if (input_size == 0)
        input_size = (uint32_t)(app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel);

    // 优先用 rknn_create_mem2 创建非缓存内存：跳过 rknn_run 前的 cache flush
    app_ctx->input_mem = rknn_create_mem2(app_ctx->rknn_ctx, input_size,
                                          RKNN_FLAG_MEMORY_NON_CACHEABLE);
    if (app_ctx->input_mem == NULL) {
        // 退回: 普通可缓存内存
        printf("[WARN] rknn_create_mem2(NON_CACHEABLE) 失败, 退回 cacheable mem\n");
        app_ctx->input_mem = rknn_create_mem(app_ctx->rknn_ctx, input_size);
    }
    if (app_ctx->input_mem == NULL) {
        printf("[WARN] rknn_create_mem(input) 失败，退回 malloc\n");
        app_ctx->input_buffer = (unsigned char*)malloc(input_size);
        if (app_ctx->input_buffer == NULL) { printf("[ERROR] malloc input buffer 也失败!\n"); return -1; }
        app_ctx->input_buffer_size = (int)input_size;
        printf("[INFO] 预分配输入缓冲区 %u 字节 (%.1f KB)，malloc 路径\n",
               input_size, input_size / 1024.0f);
    } else {
        int ret_set = rknn_set_io_mem(app_ctx->rknn_ctx, app_ctx->input_mem, &app_ctx->input_attrs[0]);
        if (ret_set != RKNN_SUCC) {
            printf("[WARN] rknn_set_io_mem(input) 失败 ret=%d，退回 malloc+rknn_inputs_set\n", ret_set);
            rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_mem);
            app_ctx->input_mem = NULL;
            app_ctx->input_buffer = (unsigned char*)malloc(input_size);
            app_ctx->input_buffer_size = (int)input_size;
        } else {
            app_ctx->input_buffer      = (unsigned char*)app_ctx->input_mem->virt_addr;
            app_ctx->input_buffer_size = (int)input_size;
            printf("[INFO] 零拷贝输入已绑定 %u 字节 (%.1f KB)，非缓存内存+跳过 cache flush\n",
                   input_size, input_size / 1024.0f);
        }
    }

    // ---------- 9. 预分配输出缓冲 (3-branch): INT8 + float32 ----------
    // 策略: rknn_outputs_get(want_float=0, is_prealloc=1) → NEON dequantize_int8
    //   - rknn_outputs_get 负责 DMA 拷贝 + cache 同步 + stride 处理 (正确可靠)
    //   - 我们只做 NEON 向量化反量化，比 RKNN 内部标量 dequant 快约 8x (~0.6ms 节省)
    //   - 不使用 rknn_set_io_mem 绑定输出（需要 RKNN_QUERY_NATIVE_OUTPUT_ATTR，避免兼容问题）
    if (app_ctx->is_3branch) {
        int total_int8_kb = 0, total_float_kb = 0;
        for (int i = 0; i < 3; ++i) {
            // INT8 prealloc 缓冲（rknn_outputs_get want_float=0 的目标）
            uint32_t int8_sz = app_ctx->output_attrs[i].n_elems * (uint32_t)sizeof(int8_t);
            app_ctx->output_int8_bufs[i] = (int8_t*)malloc(int8_sz);
            if (app_ctx->output_int8_bufs[i] != NULL) {
                app_ctx->output_int8_buf_sizes[i] = (int)int8_sz;
                total_int8_kb += (int)(int8_sz / 1024);
            }
            // float32 缓冲（NEON dequantize_int8 写入目标）
            uint32_t flt_sz = app_ctx->output_attrs[i].n_elems * (uint32_t)sizeof(float);
            app_ctx->output_float_bufs[i] = (float*)malloc(flt_sz);
            if (app_ctx->output_float_bufs[i] != NULL) {
                app_ctx->output_float_buf_sizes[i] = (int)flt_sz;
                total_float_kb += (int)(flt_sz / 1024);
            }
        }
        printf("[INFO] 预分配输出缓冲: INT8=%dKB float=%dKB，NEON dequant 路径\n",
               total_int8_kb, total_float_kb);
    }

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
    // 释放零拷贝输出 INT8 内存（必须在 rknn_destroy 之前）
    for (int i = 0; i < 3; ++i) {
        if (app_ctx->output_int8_bufs[i] != NULL) {
            free(app_ctx->output_int8_bufs[i]);
            app_ctx->output_int8_bufs[i] = NULL;
            app_ctx->output_int8_buf_sizes[i] = 0;
        }
    }
    // 释放浮点反量化缓冲（普通 malloc）
    for (int i = 0; i < 3; ++i) {
        if (app_ctx->output_float_bufs[i] != NULL) {
            free(app_ctx->output_float_bufs[i]);
            app_ctx->output_float_bufs[i] = NULL;
            app_ctx->output_float_buf_sizes[i] = 0;
        }
    }
    // 释放零拷贝输入内存（必须在 rknn_destroy 之前）
    if (app_ctx->input_mem != NULL) {
        rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_mem);
        app_ctx->input_mem = NULL;
        app_ctx->input_buffer = NULL;
        app_ctx->input_buffer_size = 0;
    } else if (app_ctx->input_buffer != NULL) {
        // fallback: 由 malloc 分配
        free(app_ctx->input_buffer);
        app_ctx->input_buffer = NULL;
        app_ctx->input_buffer_size = 0;
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
    // 优先复用预分配缓冲区，避免每帧 malloc/free
    bool input_buf_owned = false;
    if (app_ctx->input_buffer != NULL &&
        app_ctx->input_buffer_size >= static_cast<int>(dst_image.size)) {
        dst_image.virt_addr = app_ctx->input_buffer;
    } else {
        dst_image.virt_addr = (unsigned char*)malloc(dst_image.size);
        if (dst_image.virt_addr == NULL) {
            printf("[ERROR] malloc input buffer failed\n");
            if (need_free_rgb) free(rgb_buffer.virt_addr);
            return -1;
        }
        input_buf_owned = true;
    }

    // letterbox 保持宽高比不变, 用灰色(114)填充多余区域.
    // 填充值 114 是 YOLOv5 训练时的默认值, 对红外灰度图像同样适用.
    letterbox_t letterbox;
    memset(&letterbox, 0, sizeof(letterbox));
    int ret = convert_image_with_exact_letterbox(rgb_image, &dst_image, &letterbox, 114);

    // rgb_buffer 用完即可释放, 后续只用 dst_image
    if (need_free_rgb) {
        free(rgb_buffer.virt_addr);
        need_free_rgb = false;
    }
    if (ret < 0) {
        printf("[ERROR] convert_image_with_exact_letterbox failed: %d\n", ret);
        free(dst_image.virt_addr);
        return -1;
    }

    // -------- 设置输入 --------
    // 优先零拷贝路径 (input_mem 已在 init 时绑定)；
    // fallback: rknn_inputs_set（input_buffer 由 malloc 分配时）。
    if (app_ctx->input_mem == NULL) {
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
            if (input_buf_owned) free(dst_image.virt_addr);
            return -1;
        }
    }
    // (else: 零拷贝路径，letterbox 已写入 input_mem->virt_addr = input_buffer)

    // -------- NPU 推理 --------
    const double npu_t0 = now_ms();
    ret = rknn_run(app_ctx->rknn_ctx, NULL);
    app_ctx->last_npu_ms = now_ms() - npu_t0;
    if (ret != RKNN_SUCC) {
        printf("[ERROR] rknn_run failed: %d\n", ret);
        if (input_buf_owned) free(dst_image.virt_addr);
        return -1;
    }

    // -------- 获取输出 & 后处理 --------
    std::vector<Detection> decoded;

    if (app_ctx->is_3branch) {
        // want_float=0 + is_prealloc=1: RKNN 做 DMA 拷贝（含 cache/stride 处理），
        // 再用 NEON dequantize_int8 向量化反量化（比 want_float=1 内部标量快约8x）。
        rknn_output outputs[3];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < 3; ++i) {
            outputs[i].index      = i;
            outputs[i].want_float = 0;
            if (app_ctx->output_int8_bufs[i] != NULL) {
                outputs[i].is_prealloc = 1;
                outputs[i].buf         = app_ctx->output_int8_bufs[i];
                outputs[i].size        = (uint32_t)app_ctx->output_int8_buf_sizes[i];
            }
        }
        ret = rknn_outputs_get(app_ctx->rknn_ctx, 3, outputs, NULL);
        if (ret != RKNN_SUCC) {
            printf("[ERROR] rknn_outputs_get (3-branch INT8) failed: %d\n", ret);
            if (input_buf_owned) free(dst_image.virt_addr);
            return -1;
        }
        // NEON 向量化反量化: INT8 → float32
        for (int i = 0; i < 3; ++i) {
            if (app_ctx->output_float_bufs[i] != NULL && outputs[i].buf != NULL) {
                dequantize_int8(
                    (const int8_t*)outputs[i].buf,
                    app_ctx->output_float_bufs[i],
                    (int)app_ctx->output_attrs[i].n_elems,
                    app_ctx->output_attrs[i].zp,
                    app_ctx->output_attrs[i].scale);
            }
        }
        rknn_outputs_release(app_ctx->rknn_ctx, 3, outputs);
        BranchInfo branches[3];
        for (int b = 0; b < 3; ++b) {
            int idx = app_ctx->branch_index[b];
            const rknn_tensor_attr* attr = &app_ctx->output_attrs[idx];
            branches[b].data    = app_ctx->output_float_bufs[idx];
            branches[b].grid_h  = MODEL_GRID_SIZES[b];
            branches[b].grid_w  = MODEL_GRID_SIZES[b];
            branches[b].stride  = MODEL_STRIDES[b];
            branches[b].is_nchw = (attr->fmt == RKNN_TENSOR_NCHW);
        }
        decoded = decode_yolov5_3branch_output(
            branches, MODEL_NUM_ANCHORS, app_ctx->num_classes, conf_threshold);
    } else {
        // 单输出: [1, 25200, 7] 已解码
        rknn_output output;
        memset(&output, 0, sizeof(output));
        output.index = 0;
        output.want_float = 1;
        ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, &output, NULL);
        if (ret != RKNN_SUCC) {
            printf("[ERROR] rknn_outputs_get failed: %d\n", ret);
            if (input_buf_owned) free(dst_image.virt_addr);
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
    std::vector<Detection> nms_result = apply_nms(decoded, nms_threshold);

    // NMS 后二次去重: 消除 EIoU 等精确回归 loss + INT8 量化产生的近似重复框.
    // same_class_iou=0.50: 同类 IoU>=0.50 的保留高分框
    // cross_class_iou=0.70: 跨类 IoU>=0.70 (同一目标被判为不同类) 保留高分框
    *detections = dedup_detections(nms_result, 0.50f, 0.70f);

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
    if (input_buf_owned) free(dst_image.virt_addr);
    return 0;
}

// ========================================================================
//  inference_rknn_model_preloaded — 完全零拷贝快速推理路径
// ========================================================================
//
// 调用前提:
//   app_ctx->input_buffer (= input_mem->virt_addr) 已被 preprocess_to_model_buf 填充。
//
// 零拷贝流程:
//   1. preprocess_to_model_buf 写入 input_mem->virt_addr
//   2. rknn_run 前 RKNN 自动 cache flush（无需 rknn_inputs_set）
//   3. rknn_outputs_get(want_float=0) + NEON dequantize_int8 → float32
//   4. dequantize_int8 将 INT8 转 float 到 output_float_bufs[i]（NEON 加速）
//   5. decode / NMS / dedup
//
// 与旧路径对比:
//   省去 rknn_inputs_set (~1.5ms DMA 拷贝) + rknn_outputs_get (~1ms 内部拷贝)
//
int inference_rknn_model_preloaded(
    RknnAppContext* app_ctx,
    const letterbox_t* letterbox,
    int src_width,
    int src_height,
    float conf_threshold,
    float nms_threshold,
    std::vector<Detection>* detections) {
    if (app_ctx == NULL || app_ctx->input_buffer == NULL ||
        letterbox == NULL || detections == NULL) {
        return -1;
    }

    if (conf_threshold < 0.01f) conf_threshold = 0.01f;
    if (conf_threshold > 0.95f) conf_threshold = 0.95f;
    if (nms_threshold < 0.05f) nms_threshold = 0.05f;
    if (nms_threshold > 0.95f) nms_threshold = 0.95f;

    const bool debug_box = env_enabled("RKNN_DEBUG_BOX");
    detections->clear();

    // -------- NPU 推理 --------
    // 输入已由 init 时 rknn_set_io_mem 绑定到 input_mem->virt_addr，
    // rknn_run 前 RKNN 自动做 cache flush，无需 rknn_inputs_set。
    // fallback: 若零拷贝绑定失败（input_mem==NULL），退回 rknn_inputs_set。
    int ret = 0;
    if (app_ctx->input_mem == NULL) {
        rknn_input input;
        memset(&input, 0, sizeof(input));
        input.index = 0;
        input.type  = RKNN_TENSOR_UINT8;
        input.fmt   = RKNN_TENSOR_NHWC;
        input.size  = (uint32_t)(app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel);
        input.buf   = app_ctx->input_buffer;
        ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, &input);
        if (ret != RKNN_SUCC) {
            printf("[ERROR] rknn_inputs_set failed: %d\n", ret);
            return -1;
        }
    }

    const double npu_t0 = now_ms();
    ret = rknn_run(app_ctx->rknn_ctx, NULL);
    app_ctx->last_npu_ms = now_ms() - npu_t0;
    if (ret != RKNN_SUCC) {
        printf("[ERROR] rknn_run failed: %d\n", ret);
        return -1;
    }

    // -------- 获取输出 & 后处理 --------
    std::vector<Detection> decoded;

    if (app_ctx->is_3branch) {
        // want_float=0 + is_prealloc=1: RKNN 做 DMA 拷贝（含 cache/stride 处理），
        // 再用 NEON dequantize_int8 向量化反量化（比 want_float=1 内部标量快约8x）。
        rknn_output outputs[3];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < 3; ++i) {
            outputs[i].index      = i;
            outputs[i].want_float = 0;
            if (app_ctx->output_int8_bufs[i] != NULL) {
                outputs[i].is_prealloc = 1;
                outputs[i].buf         = app_ctx->output_int8_bufs[i];
                outputs[i].size        = (uint32_t)app_ctx->output_int8_buf_sizes[i];
            }
        }
        ret = rknn_outputs_get(app_ctx->rknn_ctx, 3, outputs, NULL);
        if (ret != RKNN_SUCC) {
            printf("[ERROR] rknn_outputs_get (3-branch INT8) failed: %d\n", ret);
            return -1;
        }
        // NEON 向量化反量化: INT8 → float32
        for (int i = 0; i < 3; ++i) {
            if (app_ctx->output_float_bufs[i] != NULL && outputs[i].buf != NULL) {
                dequantize_int8(
                    (const int8_t*)outputs[i].buf,
                    app_ctx->output_float_bufs[i],
                    (int)app_ctx->output_attrs[i].n_elems,
                    app_ctx->output_attrs[i].zp,
                    app_ctx->output_attrs[i].scale);
            }
        }
        rknn_outputs_release(app_ctx->rknn_ctx, 3, outputs);
        BranchInfo branches[3];
        for (int b = 0; b < 3; ++b) {
            int idx = app_ctx->branch_index[b];
            const rknn_tensor_attr* attr = &app_ctx->output_attrs[idx];
            branches[b].data    = app_ctx->output_float_bufs[idx];
            branches[b].grid_h  = MODEL_GRID_SIZES[b];
            branches[b].grid_w  = MODEL_GRID_SIZES[b];
            branches[b].stride  = MODEL_STRIDES[b];
            branches[b].is_nchw = (attr->fmt == RKNN_TENSOR_NCHW);
        }
        decoded = decode_yolov5_3branch_output(
            branches, MODEL_NUM_ANCHORS, app_ctx->num_classes, conf_threshold);
    } else {
        // 单输出路径（不常用）
        rknn_output output;
        memset(&output, 0, sizeof(output));
        output.index = 0;
        output.want_float = 1;
        ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, &output, NULL);
        if (ret != RKNN_SUCC) {
            printf("[ERROR] rknn_outputs_get failed: %d\n", ret);
            return -1;
        }
        const float* raw = (const float*)output.buf;
        decoded = decode_yolov5_output(
            raw, app_ctx->num_proposals, app_ctx->output_dim, conf_threshold);
        rknn_outputs_release(app_ctx->rknn_ctx, 1, &output);
    }

    // 坐标映射 + NMS
    const int decoded_count = static_cast<int>(decoded.size());
    scale_detections(decoded, *letterbox, src_width, src_height);
    std::vector<Detection> nms_result = apply_nms(decoded, nms_threshold);
    *detections = dedup_detections(nms_result, 0.50f, 0.70f);

    if (debug_box) {
        static int dbg = 0;
        if (dbg < 20) {
            printf("[DEBUG][preloaded] frame=%d decoded=%d nms_out=%d\n",
                   dbg, decoded_count, static_cast<int>(detections->size()));
            print_debug_detections(*detections, "post_nms_top", 8);
        }
        ++dbg;
    }

    return 0;
}
