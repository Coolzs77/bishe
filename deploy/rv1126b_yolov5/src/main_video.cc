// ========================================================================
//  板端红外视频检测程序
// ========================================================================
//
// 用法:
//   # 红外视频文件检测
//   ./bishe_rknn_video <model.rknn> <红外视频.mp4> [标签文件] [输出视频] [conf] [nms]
//
//   # 红外摄像头实时检测 (后期接上红外摄像头时使用)
//   ./bishe_rknn_video <model.rknn> /dev/video0 [标签文件] [输出视频] [conf] [nms]
//   ./bishe_rknn_video <model.rknn> camera:0    [标签文件] [输出视频] [conf] [nms]
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

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
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
    for (int i = 0; i < line_count; ++i) {
        if (lines[i] != NULL && strlen(lines[i]) > 0)
            labels.push_back(lines[i]);
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

// ---- cv::Mat → image_buffer_t 转换 ----
// OpenCV 解码视频帧为 BGR, 这里转为 RGB888 供 RKNN 推理使用.
// 调用者负责释放 out->virt_addr.
int mat_to_image_buffer(const cv::Mat& frame, image_buffer_t* out) {
    if (frame.empty() || out == NULL) return -1;

    cv::Mat rgb;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    } else if (frame.channels() == 1) {
        // 单通道灰度 (某些红外摄像头直接输出灰度)
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("红外视频检测 - 板端推理程序\n\n");
        printf("用法:\n");
        printf("  视频文件: %s <model.rknn> <视频.mp4> [标签] [输出视频] [conf] [nms]\n", argv[0]);
        printf("  摄像头:   %s <model.rknn> camera:0   [标签] [输出视频] [conf] [nms]\n", argv[0]);
        printf("  摄像头:   %s <model.rknn> /dev/video0 [标签] [输出视频] [conf] [nms]\n", argv[0]);
        return -1;
    }

    const char* model_path  = argv[1];
    const char* input_path  = argv[2];
    const char* labels_path = argc >= 4 ? argv[3] : "./model/infrared_labels.txt";
    const char* output_path = argc >= 5 ? argv[4] : "out_video.mp4";
    float conf_threshold    = argc >= 6 ? static_cast<float>(atof(argv[5])) : 0.35f;
    float nms_threshold     = argc >= 7 ? static_cast<float>(atof(argv[6])) : 0.40f;

    printf("=== 红外视频检测 ===\n");
    printf("  模型:   %s\n", model_path);
    printf("  输入:   %s\n", input_path);
    printf("  标签:   %s\n", labels_path);
    printf("  输出:   %s\n", output_path);
    printf("  conf:   %.2f\n", conf_threshold);
    printf("  nms:    %.2f\n", nms_threshold);
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
    // 使用 mp4v 编码器 (MPEG-4), 板子上通常可用
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer.open(output_path, fourcc, fps, cv::Size(frame_w, frame_h), true);
    if (!writer.isOpened()) {
        printf("[WARN] 无法创建输出视频: %s (仅打印结果, 不保存视频)\n", output_path);
    }

    // ---- 逐帧推理循环 ----
    cv::Mat frame;
    int frame_count = 0;
    int total_detections = 0;
    double total_infer_ms = 0.0;
    double loop_start = get_time_ms();

    while (cap.read(frame)) {
        // 把 cv::Mat (BGR) 转为 image_buffer_t (RGB888)
        image_buffer_t img_buf;
        if (mat_to_image_buffer(frame, &img_buf) != 0) {
            printf("[WARN] 帧 %d 转换失败, 跳过\n", frame_count);
            ++frame_count;
            continue;
        }

        // RKNN 推理
        double t0 = get_time_ms();
        std::vector<Detection> detections;
        int ret = inference_rknn_model(&app_ctx, &img_buf, conf_threshold, nms_threshold, &detections);
        double infer_ms = get_time_ms() - t0;

        free(img_buf.virt_addr);

        if (ret != 0) {
            printf("[WARN] 帧 %d 推理失败, 跳过\n", frame_count);
            ++frame_count;
            continue;
        }

        total_infer_ms += infer_ms;
        total_detections += static_cast<int>(detections.size());

        // 在原始 BGR 帧上绘制检测结果
        draw_detections_on_mat(frame, detections, labels);

        // 绘制帧信息 (帧号 + 推理耗时 + 检测数)
        char info[128];
        snprintf(info, sizeof(info), "F:%d Det:%d %.1fms",
                 frame_count, static_cast<int>(detections.size()), infer_ms);
        cv::putText(frame, info, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        // 写入输出视频
        if (writer.isOpened()) {
            writer.write(frame);
        }

        ++frame_count;

        // 每 100 帧打印一次进度
        if (frame_count % 100 == 0) {
            double avg_ms = total_infer_ms / frame_count;
            if (!from_camera && total_frames > 0) {
                printf("  进度: %d/%d 帧 (%.0f%%), 平均推理: %.1f ms, NPU FPS: %.1f\n",
                       frame_count, total_frames,
                       100.0 * frame_count / total_frames,
                       avg_ms, 1000.0 / avg_ms);
            } else {
                printf("  已处理: %d 帧, 平均推理: %.1f ms, NPU FPS: %.1f\n",
                       frame_count, avg_ms, 1000.0 / avg_ms);
            }
        }
    }

    double total_ms = get_time_ms() - loop_start;

    // ---- 汇总输出 ----
    printf("\n=== 推理完成 ===\n");
    printf("  总帧数:     %d\n", frame_count);
    printf("  总检测数:   %d\n", total_detections);
    if (frame_count > 0) {
        printf("  平均推理:   %.1f ms/帧\n", total_infer_ms / frame_count);
        printf("  NPU FPS:    %.1f\n", 1000.0 * frame_count / total_infer_ms);
        printf("  端到端 FPS: %.1f (含视频读写)\n", 1000.0 * frame_count / total_ms);
    }
    if (writer.isOpened()) {
        printf("  输出视频:   %s\n", output_path);
    }

    // ---- 释放资源 ----
    cap.release();
    writer.release();
    release_rknn_model(&app_ctx);
    return 0;
}
