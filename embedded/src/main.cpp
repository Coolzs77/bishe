/**
 * @file main.cpp
 * @brief 红外目标检测与跟踪系统主程序
 * @author 张仕卓
 * @date 2026
 */

#include <iostream>
#include <string>
#include <signal.h>
#include "pipeline.h"

// 全局运行标志
volatile bool g_running = true;

/**
 * @brief 信号处理函数
 */
void signalHandler(int signum) {
    std::cout << "\n接收到信号 " << signum << "，正在退出..." << std::endl;
    g_running = false;
}

/**
 * @brief 打印使用说明
 */
void printUsage(const char* program) {
    std::cout << "用法: " << program << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --model <path>    RKNN模型路径" << std::endl;
    std::cout << "  --config <path>   配置文件路径" << std::endl;
    std::cout << "  --source <path>   视频源 (设备路径或文件)" << std::endl;
    std::cout << "  --help            显示帮助信息" << std::endl;
}

/**
 * @brief 结果回调函数
 */
void onResult(const FrameResult& result) {
    std::cout << "帧 " << result.frame_id 
              << " | 目标数: " << result.objects.size()
              << " | 总耗时: " << result.total_time << " ms"
              << " | FPS: " << (1000.0f / result.total_time)
              << std::endl;
    
    // 打印每个跟踪目标
    for (const auto& obj : result.objects) {
        std::cout << "  ID=" << obj.id 
                  << " 类别=" << obj.class_id
                  << " 位置=(" << obj.box.x << "," << obj.box.y << ")"
                  << " 尺寸=" << obj.box.width << "x" << obj.box.height
                  << std::endl;
    }
}

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  红外目标检测与跟踪系统" << std::endl;
    std::cout << "  版本: 1.0.0" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 默认参数
    std::string model_path = "models/best.rknn";
    std::string config_path = "configs/deploy.yaml";
    std::string source = "/dev/video0";
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--source" && i + 1 < argc) {
            source = argv[++i];
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    std::cout << "\n配置:" << std::endl;
    std::cout << "  模型: " << model_path << std::endl;
    std::cout << "  配置: " << config_path << std::endl;
    std::cout << "  视频源: " << source << std::endl;
    
    // 注册信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // 创建并初始化流水线
    Pipeline pipeline(config_path);
    
    if (!pipeline.init()) {
        std::cerr << "初始化流水线失败!" << std::endl;
        return -1;
    }
    
    // 设置结果回调
    pipeline.setResultCallback(onResult);
    
    // 开始处理视频流
    std::cout << "\n开始处理视频流..." << std::endl;
    pipeline.startVideoStream(source);
    
    // 主循环
    while (g_running) {
        // 等待一段时间
        usleep(100000);  // 100ms
    }
    
    // 停止视频流
    pipeline.stopVideoStream();
    
    // 打印统计信息
    std::cout << "\n" << pipeline.getStats() << std::endl;
    
    std::cout << "程序已退出" << std::endl;
    return 0;
}
