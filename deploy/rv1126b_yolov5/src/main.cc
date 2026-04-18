#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

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
//   ./bishe_rknn_detect <model.rknn> [红外图片|图片目录] [标签文件] [输出图片|输出目录] [置信度] [NMS]
//
// 说明:
//   本程序接受红外热成像图片 (JPEG/PNG 格式, 图像内容为红外灰度),
//   使用 RKNN NPU 执行 YOLOv5 推理, 检测 person 和 car 两类目标,
//   然后在图像上绘制检测框并保存结果图。若不传图片路径, 默认批量处理
//   ./model 下所有 test_*.jpg/png/bmp 测试图片并输出到 ./outputs.
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

bool is_directory_path(const char* path) {
    struct stat st;
    if (path == NULL || stat(path, &st) != 0) {
        return false;
    }
    return S_ISDIR(st.st_mode);
}

bool is_regular_file_path(const char* path) {
    struct stat st;
    if (path == NULL || stat(path, &st) != 0) {
        return false;
    }
    return S_ISREG(st.st_mode);
}

std::string to_lower_ascii(std::string text) {
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] >= 'A' && text[i] <= 'Z') {
            text[i] = static_cast<char>(text[i] - 'A' + 'a');
        }
    }
    return text;
}

bool has_image_extension(const std::string& name) {
    const std::size_t dot = name.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }
    const std::string ext = to_lower_ascii(name.substr(dot + 1));
    return ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp";
}

// EIoU 在 INT8 板端更容易出现“阈值附近候选偏多”，默认提高 conf 抑制碎框。
bool is_eiou_model_path(const char* model_path) {
    if (model_path == NULL) {
        return false;
    }
    const std::string lower = to_lower_ascii(std::string(model_path));
    return lower.find("best_eiou") != std::string::npos ||
           lower.find("exp07_eiou") != std::string::npos;
}

std::string file_name_from_path(const std::string& path) {
    const std::size_t p = path.find_last_of("/\\");
    if (p == std::string::npos) {
        return path;
    }
    return path.substr(p + 1);
}

std::string file_stem_from_path(const std::string& path) {
    const std::string name = file_name_from_path(path);
    const std::size_t dot = name.find_last_of('.');
    if (dot == std::string::npos) {
        return name;
    }
    return name.substr(0, dot);
}

std::string join_path(const std::string& dir_path, const std::string& name) {
    if (dir_path.empty()) {
        return name;
    }
    if (dir_path[dir_path.size() - 1] == '/') {
        return dir_path + name;
    }
    return dir_path + "/" + name;
}

bool ensure_output_dir(const char* dir_path) {
    if (dir_path == NULL || dir_path[0] == '\0') {
        return false;
    }
    if (is_directory_path(dir_path)) {
        return true;
    }
    if (mkdir(dir_path, 0755) == 0) {
        return true;
    }
    if (errno == EEXIST) {
        return is_directory_path(dir_path);
    }
    return false;
}

void collect_images_from_dir(
    const std::string& dir_path,
    bool prefer_test_prefix,
    std::vector<std::string>* image_paths) {
    image_paths->clear();
    DIR* dir = opendir(dir_path.c_str());
    if (dir == NULL) {
        return;
    }

    std::vector<std::string> preferred;
    std::vector<std::string> fallback;
    struct dirent* ent = NULL;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name == NULL) continue;
        const std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        if (!has_image_extension(name)) continue;

        const std::string full_path = join_path(dir_path, name);
        if (!is_regular_file_path(full_path.c_str())) continue;

        const std::string lower_name = to_lower_ascii(name);
        if (prefer_test_prefix && lower_name.find("test_") == 0) {
            preferred.push_back(full_path);
        }
        fallback.push_back(full_path);
    }
    closedir(dir);

    std::sort(preferred.begin(), preferred.end());
    std::sort(fallback.begin(), fallback.end());
    if (prefer_test_prefix && !preferred.empty()) {
        *image_paths = preferred;
    } else {
        *image_paths = fallback;
    }
}

int run_single_image(
    RknnAppContext* app_ctx,
    const std::vector<std::string>& labels,
    const char* image_path,
    const char* output_path,
    float conf_threshold,
    float nms_threshold) {
    // 从磁盘读取红外图片.
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(src_image));
    if (read_image(image_path, &src_image) != 0) {
        printf("[ERROR] 读取红外图片失败: %s\n", image_path);
        return -1;
    }

    printf("红外图片已加载: %s (%dx%d, format=%d)\n",
           image_path, src_image.width, src_image.height, src_image.format);

    // 执行完整推理 (预处理 → NPU → 后处理).
    std::vector<Detection> detections;
    if (inference_rknn_model(app_ctx, &src_image, conf_threshold, nms_threshold, &detections) != 0) {
        printf("[ERROR] 推理失败: %s\n", image_path);
        if (src_image.virt_addr != NULL) {
            free(src_image.virt_addr);
        }
        return -1;
    }

    printf("检测到 %d 个目标:\n", static_cast<int>(detections.size()));

    // image_drawing 仅支持 RGB888，灰度输入时先转为 3 通道用于可视化输出。
    image_buffer_t rgb_image;
    bool need_free_rgb = false;
    image_buffer_t* draw_image = ensure_rgb888(&src_image, &rgb_image, &need_free_rgb);
    if (draw_image == NULL) {
        printf("[ERROR] 图像可视化前 RGB 转换失败: %s\n", image_path);
        if (src_image.virt_addr != NULL) {
            free(src_image.virt_addr);
        }
        return -1;
    }

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

    if (write_image(output_path, draw_image) == 0) {
        printf("[ERROR] 保存结果图失败: %s\n", output_path);
        if (need_free_rgb && rgb_image.virt_addr != NULL) {
            free(rgb_image.virt_addr);
        }
        if (src_image.virt_addr != NULL) {
            free(src_image.virt_addr);
        }
        return -1;
    }
    printf("结果已保存: %s\n\n", output_path);

    if (need_free_rgb && rgb_image.virt_addr != NULL) {
        free(rgb_image.virt_addr);
    }
    if (src_image.virt_addr != NULL) {
        free(src_image.virt_addr);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // 命令行参数:
    //   argv[1]: model.rknn    RKNN 模型文件
    //   argv[2]: image_or_dir  红外图片路径或图片目录 (可选, 默认 ./model)
    //   argv[3]: labels.txt    标签文件 (可选, 默认 ./model/infrared_labels.txt)
    //   argv[4]: output_path   输出图片路径或输出目录 (可选, 默认单图 out.png / 批量 ./outputs)
    //   argv[5]: conf_thresh   置信度阈值 (可选, 默认 0.25)
    //   argv[6]: nms_thresh    NMS 阈值 (可选, 默认 0.45)
    if (argc < 2) {
        printf("红外目标检测 - 板端推理程序\n");
        printf("用法: %s <model.rknn> [红外图片|图片目录] [标签文件] [输出图片|输出目录] [置信度] [NMS阈值]\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    const char* image_or_dir = argc >= 3 ? argv[2] : "./model";
    const bool batch_mode = (argc < 3) || is_directory_path(image_or_dir);
    const bool default_batch_mode = (argc < 3);
    const char* labels_path = argc >= 4 ? argv[3] : "./model/infrared_labels.txt";
    const char* output_path = argc >= 5 ? argv[4] : (batch_mode ? "./outputs" : "out.png");
    const float default_conf_threshold = is_eiou_model_path(model_path) ? 0.55f : 0.25f;
    const float default_nms_threshold = is_eiou_model_path(model_path) ? 0.35f : 0.45f;
    const float conf_threshold = argc >= 6 ? static_cast<float>(atof(argv[5])) : default_conf_threshold;
    const float nms_threshold = argc >= 7 ? static_cast<float>(atof(argv[6])) : default_nms_threshold;

    if (!batch_mode && !is_regular_file_path(image_or_dir)) {
        printf("[ERROR] 输入路径不存在或不是文件: %s\n", image_or_dir);
        return -1;
    }
    if (batch_mode && !is_directory_path(image_or_dir)) {
        printf("[ERROR] 输入目录不存在: %s\n", image_or_dir);
        return -1;
    }
    if (batch_mode && !ensure_output_dir(output_path)) {
        printf("[ERROR] 无法创建输出目录: %s\n", output_path);
        return -1;
    }

    printf("=== 红外目标检测 ===\n");
    printf("  模型:   %s\n", model_path);
    printf("  模式:   %s\n", batch_mode ? "批量图片" : "单图");
    printf("  输入:   %s\n", image_or_dir);
    printf("  标签:   %s\n", labels_path);
    printf("  输出:   %s%s\n", output_path, batch_mode ? " (目录)" : "");
    printf("  conf:   %.2f\n", conf_threshold);
    printf("  nms:    %.2f\n", nms_threshold);
    if (argc < 6 && is_eiou_model_path(model_path)) {
        printf("  note:   EIoU 自动: conf=0.55, nms=0.35 (可用第5/6参数覆盖)\n");
    }
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

    std::vector<std::string> image_paths;
    if (batch_mode) {
        collect_images_from_dir(image_or_dir, default_batch_mode, &image_paths);
        if (image_paths.empty()) {
            printf("[ERROR] 输入目录中未找到可推理图片: %s\n", image_or_dir);
            release_rknn_model(&app_ctx);
            return -1;
        }
        printf("批量推理图片数: %d\n\n", static_cast<int>(image_paths.size()));
    } else {
        image_paths.push_back(image_or_dir);
    }

    int success_count = 0;
    int fail_count = 0;
    for (std::size_t i = 0; i < image_paths.size(); ++i) {
        const std::string& image_path = image_paths[i];
        std::string out_path = output_path;
        if (batch_mode) {
            out_path = join_path(output_path, file_stem_from_path(image_path) + "_out.png");
        }
        if (run_single_image(&app_ctx, labels, image_path.c_str(), out_path.c_str(), conf_threshold, nms_threshold) == 0) {
            ++success_count;
        } else {
            ++fail_count;
        }
    }

    if (batch_mode) {
        printf("批量完成: success=%d fail=%d\n", success_count, fail_count);
    }
    release_rknn_model(&app_ctx);
    return fail_count == 0 ? 0 : -1;
}
