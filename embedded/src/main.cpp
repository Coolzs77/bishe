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
    std::cout << "\nReceived signal " << signum << ", exiting..." << std::endl;
    g_running = false;
}

/**
 * @brief 打印使用说明
 */
void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --model <path>    RKNN model path" << std::endl;
    std::cout << "  --config <path>   Configuration file path" << std::endl;
    std::cout << "  --source <path>   Video source (device path or file)" << std::endl;
    std::cout << "  --help            Show help information" << std::endl;
}

/**
 * @brief 结果回调函数
 */
void onResult(const FrameResult& result) {
    std::cout << "Frame " << result.frame_id 
              << " | Objects: " << result.objects.size()
              << " | Total time: " << result.total_time << " ms"
              << " | FPS: " << (1000.0f / result.total_time)
              << std::endl;
    
    // 打印每个跟踪目标
    for (const auto& obj : result.objects) {
        std::cout << "  ID=" << obj.id 
                  << " Class=" << obj.class_id
                  << " Position=(" << obj.box.x << "," << obj.box.y << ")"
                  << " Size=" << obj.box.width << "x" << obj.box.height
                  << std::endl;
    }
}

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Infrared Object Detection & Tracking System" << std::endl;
    std::cout << "  Version: 1.0.0" << std::endl;
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
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Model: " << model_path << std::endl;
    std::cout << "  Config: " << config_path << std::endl;
    std::cout << "  Video source: " << source << std::endl;
    
    // 注册信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // 创建并初始化流水线
    Pipeline pipeline(config_path);
    
    if (!pipeline.init()) {
        std::cerr << "Pipeline initialization failed!" << std::endl;
        return -1;
    }
    
    // 设置结果回调
    pipeline.setResultCallback(onResult);
    
    // 开始处理视频流
    std::cout << "\nStarting video stream processing..." << std::endl;
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
    
    std::cout << "Program exited" << std::endl;
    return 0;
}
