#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "file_utils.h"
#include "image_drawing.h"
#include "image_utils.h"
#include "rknn_detector.hpp"

// ========================================================================
//  板端红外目标检测主程序
// ========================================================================
//
// 用法:
//   ./bishe_rknn_detect <model.rknn> <红外图片> [标签文件] [输出图片] [置信度] [NMS]
//
// 说明:
//   本程序接受红外热成像图片 (JPEG/PNG 格式, 图像内容为红外灰度),
//   使用 RKNN NPU 执行 YOLOv5 推理, 检测 person 和 car 两类目标,
//   然后在图像上绘制检测框并保存结果图.
//
//   红外图像虽然看起来是灰度的, 但 JPEG 文件会被 read_image 函数
//   自动读取为 3 通道 RGB888 (三通道值相同). 这和训练时 OpenCV 的
//   行为一致, 所以无需额外处理.
//
//   如果你从红外摄像头获取到的是原始单通道灰度帧, 推理函数
//   inference_rknn_model 中的 ensure_rgb888 会自动转为 3 通道.
// ========================================================================

namespace {

// 读取外部标签文件.
// 每行一个类别名, 例如:
//   person
//   car
std::vector<std::string> load_labels(const char* path) {
    std::vector<std::string> labels;
    int line_count = 0;
    char** lines = read_lines_from_file(path, &line_count);
    if (lines == NULL) {
        return labels;
    }

    for (int index = 0; index < line_count; ++index) {
        if (lines[index] != NULL && strlen(lines[index]) > 0) {
            labels.push_back(lines[index]);
        }
    }
    free_lines(lines, line_count);
    return labels;
}

// 根据类别编号返回类别名.
const char* class_name_from_id(const std::vector<std::string>& labels, int class_id) {
    if (class_id < 0 || class_id >= static_cast<int>(labels.size())) {
        return "unknown";
    }
    return labels[class_id].c_str();
}

// 与项目视频风格保持一致: 统一红色框.
unsigned int color_from_class(int class_id) {
    static const unsigned int colors[] = {
        COLOR_BLUE,
        COLOR_GREEN,
        COLOR_RED,
        COLOR_YELLOW,
        COLOR_ORANGE,
        COLOR_WHITE,
    };
    int n = static_cast<int>(sizeof(colors) / sizeof(colors[0]));
    return colors[class_id % n];
}

}  // namespace

int main(int argc, char** argv) {
    // 命令行参数:
    //   argv[1]: model.rknn    RKNN 模型文件
    //   argv[2]: image_path    红外图片路径 (JPEG/PNG)
    //   argv[3]: labels.txt    标签文件 (可选, 默认 ./model/infrared_labels.txt)
    //   argv[4]: output_path   输出图片路径 (可选, 默认 out.png)
    //   argv[5]: conf_thresh   置信度阈值 (可选, 默认 0.25)
    //   argv[6]: nms_thresh    NMS 阈值 (可选, 默认 0.45)
    if (argc < 3) {
        printf("红外目标检测 - 板端推理程序\n");
        printf("用法: %s <model.rknn> <红外图片> [标签文件] [输出图片] [置信度] [NMS阈值]\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];
    const char* labels_path = argc >= 4 ? argv[3] : "./model/infrared_labels.txt";
    const char* output_path = argc >= 5 ? argv[4] : "out.png";
    const float conf_threshold = argc >= 6 ? static_cast<float>(atof(argv[5])) : 0.35f;
    const float nms_threshold = argc >= 7 ? static_cast<float>(atof(argv[6])) : 0.40f;

    printf("=== 红外目标检测 ===\n");
    printf("  模型:   %s\n", model_path);
    printf("  图片:   %s\n", image_path);
    printf("  标签:   %s\n", labels_path);
    printf("  输出:   %s\n", output_path);
    printf("  conf:   %.2f\n", conf_threshold);
    printf("  nms:    %.2f\n", nms_threshold);
    printf("\n");

    // 加载类别名, 后面打印日志和画框都依赖这一步.
    std::vector<std::string> labels = load_labels(labels_path);
    if (labels.empty()) {
        printf("[ERROR] 加载标签文件失败: %s\n", labels_path);
        return -1;
    }

    // 初始化 RKNN 运行时和模型信息.
    RknnAppContext app_ctx;
    if (init_rknn_model(model_path, &app_ctx) != 0) {
        printf("[ERROR] 初始化 RKNN 模型失败\n");
        return -1;
    }

    // 校验标签文件和模型类别数是否匹配.
    if (app_ctx.num_classes != static_cast<int>(labels.size())) {
        printf("[WARN] 标签数量 (%d) != 模型类别数 (%d)\n",
               static_cast<int>(labels.size()), app_ctx.num_classes);
    }

    // 从磁盘读取红外图片.
    // read_image 会根据文件扩展名 (JPEG/PNG) 自动解码.
    // 即使原始红外图像是灰度的, JPEG 解码也会输出 RGB888 (三通道值相同).
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(src_image));
    if (read_image(image_path, &src_image) != 0) {
        printf("[ERROR] 读取红外图片失败: %s\n", image_path);
        release_rknn_model(&app_ctx);
        return -1;
    }
    printf("红外图片已加载: %dx%d, format=%d\n",
           src_image.width, src_image.height, src_image.format);

    // 执行完整推理 (预处理 → NPU → 后处理).
    // 推理函数内部会自动处理灰度→3通道转换.
    std::vector<Detection> detections;
    if (inference_rknn_model(&app_ctx, &src_image, conf_threshold, nms_threshold, &detections) != 0) {
        printf("[ERROR] 推理失败\n");
        if (src_image.virt_addr != NULL) {
            free(src_image.virt_addr);
        }
        release_rknn_model(&app_ctx);
        return -1;
    }

    printf("\n检测到 %d 个目标:\n", static_cast<int>(detections.size()));

    // image_drawing 仅支持 RGB888，灰度输入时先转为 3 通道用于可视化输出。
    image_buffer_t rgb_image;
    bool need_free_rgb = false;
    image_buffer_t* draw_image = ensure_rgb888(&src_image, &rgb_image, &need_free_rgb);
    if (draw_image == NULL) {
        printf("[ERROR] 图像可视化前 RGB 转换失败\n");
        if (src_image.virt_addr != NULL) {
            free(src_image.virt_addr);
        }
        release_rknn_model(&app_ctx);
        return -1;
    }

    // 在红外图像上绘制检测框和类别文字.
    char text[256];
    for (std::size_t index = 0; index < detections.size(); ++index) {
        const Detection& det = detections[index];
        const int x1 = static_cast<int>(det.x1);
        const int y1 = static_cast<int>(det.y1);
        const int x2 = static_cast<int>(det.x2);
        const int y2 = static_cast<int>(det.y2);
        const char* class_name = class_name_from_id(labels, det.class_id);

        printf("  [%d] %s @ (%d,%d)-(%d,%d) %.1f%%\n",
               static_cast<int>(index), class_name, x1, y1, x2, y2, det.score * 100.0f);
        draw_rectangle(draw_image, x1, y1, x2 - x1, y2 - y1, color_from_class(det.class_id), 3);
        snprintf(text, sizeof(text), "%s %.1f%%", class_name, det.score * 100.0f);
        draw_text(draw_image, text, x1, y1 > 20 ? y1 - 20 : y1 + 10, COLOR_WHITE, 10);
    }

    // 保存检测结果图.
    // 注意: write_image 内部调用 stbi_write_png, 返回非零表示成功, 0 表示失败.
    if (write_image(output_path, draw_image) == 0) {
        printf("[ERROR] 保存结果图失败: %s\n", output_path);
    } else {
        printf("\n结果已保存: %s\n", output_path);
    }

    // 收尾: 释放图片内存和 RKNN 资源.
    if (need_free_rgb && rgb_image.virt_addr != NULL) {
        free(rgb_image.virt_addr);
    }
    if (src_image.virt_addr != NULL) {
        free(src_image.virt_addr);
    }
    release_rknn_model(&app_ctx);
    return 0;
}