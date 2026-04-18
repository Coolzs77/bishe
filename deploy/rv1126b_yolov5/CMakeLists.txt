# ========================================================================
#  CMakeLists.txt — 板端推理可执行文件的构建配置
# ========================================================================
#
#  构建结果:
#    bishe_rknn_detect — 红外图片检测 (不依赖 OpenCV)
#    bishe_rknn_video  — 红外视频检测 / 摄像头实时推理 (依赖 OpenCV)
#
#  依赖来源: rknn_model_zoo 的 3rdparty (librknnrt, librga 等)
#             和 utils (fileutils, imageutils, imagedrawing)
#  使用方式: 一般不直接调用, 而是通过 build_rv1126b.sh 脚本间接调用
# ========================================================================

# 最低要求 CMake 3.15, 因为用到了 cmake --install 和 target_link_libraries 的现代语法.
cmake_minimum_required(VERSION 3.15)

# 定义项目名称和支持的语言 (C 和 C++).
project(bishe_rknn_yolov5 C CXX)

# C 标准: C11; C++ 标准: C++11 (RKNN SDK 对 C++11 兼容性最好).
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)    # 强制要求编译器支持 C++11, 不能回退
set(CMAKE_CXX_EXTENSIONS OFF)           # 禁用 GNU 扩展, 保持跨平台兼容

# ---------- 目标芯片平台 ----------
# 默认 rv1126b; 如果你换了板子, 可在 cmake 命令行用 -DTARGET_SOC=xxx 覆盖.
if(NOT DEFINED TARGET_SOC)
    set(TARGET_SOC rv1126b)
endif()

# ---------- 可选开关 ----------
# ENABLE_ASAN:  开启 Address Sanitizer, 用于调试内存越界/野指针问题.
# DISABLE_RGA:  禁用 RGA 硬件加速图像缩放 (CI 或无 RGA 的板子可用).
# DISABLE_LIBJPEG: 禁用 libjpeg, 改用 stb_image.
option(ENABLE_ASAN "Build with address sanitizer" OFF)
option(DISABLE_RGA "Disable RGA" OFF)
option(DISABLE_LIBJPEG "Disable libjpeg" OFF)

# ---------- 发布构建优化标志 (仅 Release/无明确 BuildType 时生效) ----------
# 根据目标架构自动选择合适标志:
#   AArch64 (aarch64-linux-gnu-*): NEON 始终启用, 无需 -mfpu/-mfloat-abi
#   ARMv7   (arm-linux-gnueabihf-*): 需要显式指定 NEON + hard-float
# -O3:          最高 GCC 优化 (循环展开 + 向量化)
# -ffast-math:  允许 NaN/Inf 放宽 + 关联重排, sigmoid/logit 等大幅提速
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        # AArch64: NEON 内建, 无需额外 -mfpu/-mfloat-abi
        set(OPT_FLAGS "-O3 -march=armv8-a+simd -ffast-math")
    else()
        # ARMv7 (arm-linux-gnueabihf)
        set(OPT_FLAGS "-O3 -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -ffast-math")
    endif()
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OPT_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_FLAGS}")
    message(STATUS "Release 优化标志: ${OPT_FLAGS}")
endif()

# 如果开启 ASAN, 给编译和链接加上对应 flag.
if(ENABLE_ASAN)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

# ---------- RKNN Model Zoo 路径 ----------
# 相对于本 CMakeLists.txt 向上两层就是仓库根目录,
# 再进入 rknn_model_zoo.
# 你的工程目录结构:
#   bishe/
#     rknn_model_zoo/       ← 第三方 SDK
#     deploy/rv1126b_yolov5/ ← 本目录
set(RKNN_MODEL_ZOO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../rknn_model_zoo)

# 引入 3rdparty: 它会导入 LIBRKNNRT (编译时链接 librknnrt.so)
# 以及 librga 等, 并设置 LIBRKNNRT_INCLUDES.
add_subdirectory(${RKNN_MODEL_ZOO_ROOT}/3rdparty ${CMAKE_BINARY_DIR}/rknn_3rdparty.out)

# ---------- 覆盖 RGA: 使用从板子 adb pull 的 librga.so ----------
# rknn_model_zoo 自带的 librga.a 版本 (v1.10.1) 与板子内核 RGA 驱动 (v4.x) 不兼容,
# 运行时会报 "rga2 get info failed" 然后回退 CPU.
# 解决方案: 用 adb pull 拉取板子上已安装的 librga.so (与板子内核驱动版本匹配),
# 动态链接代替静态 librga.a.
set(RGA_BOARD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga_board)
if(EXISTS ${RGA_BOARD_DIR}/librga.so)
    message(STATUS "使用板子 RGA 库: ${RGA_BOARD_DIR}/librga.so (替代 rknn_model_zoo 旧版)")
    set(LIBRGA ${RGA_BOARD_DIR}/librga.so)
    install(PROGRAMS ${RGA_BOARD_DIR}/librga.so DESTINATION lib)
else()
    message(WARNING "未找到板子 RGA 库 (3rdparty/rga_board/librga.so).\n"
                    "  预处理将回退 CPU, 性能不受影响但会输出 RGA 错误日志.\n"
                    "  请参考 README.md 用 adb pull 拉取板子的 librga.so.")
endif()

# 引入 utils: fileutils (file_utils.h), imageutils (image_utils.h),
# imagedrawing (image_drawing.h) 等工具库.
add_subdirectory(${RKNN_MODEL_ZOO_ROOT}/utils ${CMAKE_BINARY_DIR}/rknn_utils.out)

# 关闭第三方 utils 的编译警告 (它们来自 rknn_model_zoo, 不属于本项目代码).
foreach(_tgt fileutils imageutils imagedrawing audioutils)
    if(TARGET ${_tgt})
        target_compile_options(${_tgt} PRIVATE -w)
    endif()
endforeach()

# 设置 RPATH, 让可执行文件运行时从同目录的 lib/ 加载 .so 动态库.
# 这样部署时只需要把 lib/ 和可执行文件放在一起就行.
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

# ---------- 构建目标 ----------
# bishe_rknn_detect: 板端推理主程序.
# 源文件:
#   main.cc          — 入口, 命令行解析, 可视化输出
#   rknn_detector.cc — RKNN C API 封装 (加载 / 推理 / 释放)
#   postprocess.cc   — 后处理: 解码 [1,25200,7] + 坐标映射 + NMS
add_executable(bishe_rknn_detect
    src/main.cc
    src/rknn_detector.cc
    src/postprocess.cc
)

# 头文件搜索路径:
#   src/           — rknn_detector.hpp, postprocess.hpp
#   LIBRKNNRT_INCLUDES — rknn_api.h, rknn_matmul_api.h ...
#   utils/         — file_utils.h, image_utils.h, image_drawing.h
target_include_directories(bishe_rknn_detect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LIBRKNNRT_INCLUDES}
    ${RKNN_MODEL_ZOO_ROOT}/utils
)

# 链接库:
#   fileutils    — 读取文件到内存 (read_data_from_file, read_lines_from_file 等)
#   imageutils   — 图像读写和 letterbox 变换
#   imagedrawing — 在图像上画框/写字
#   LIBRKNNRT    — librknnrt.so, RKNN C API 的运行时库
#   dl           — dlopen/dlsym, RKNN 内部会用到
target_link_libraries(bishe_rknn_detect
    fileutils
    imageutils
    imagedrawing
    ${LIBRKNNRT}
    dl
)

# 板子的 librga.so 用 GLIBC 2.34 / GLIBCXX 3.4.29 编译,
# Ubuntu 20.04 交叉工具链只有 GLIBC 2.31, 链接时会报 undefined reference.
# --allow-shlib-undefined 跳过编译时检查; 运行时在板子上由系统动态链接器解析.
target_link_options(bishe_rknn_detect PRIVATE
    "LINKER:--allow-shlib-undefined"
)

# Linux 下需要显式链接 pthread, 因为 RKNN 和 RGA 内部用了多线程.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(bishe_rknn_detect Threads::Threads)
endif()

# ---------- 安装规则 ----------
# cmake --install 时会把可执行文件和模型文件复制到安装目录.
# 部署时直接把安装目录整个拷贝到板子即可.
install(TARGETS bishe_rknn_detect DESTINATION .)

# 复制标签文件 (从 model/ 目录).
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/infrared_labels.txt DESTINATION model)

# 复制所有 .rknn 模型文件 (如果你已经把转换好的 .rknn 放到了 model/ 里).
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)

# ---------- 视频检测可执行文件 (依赖板子的 OpenCV videoio) ----------
# 板子上 apt 安装了完整的 OpenCV 4.6.0 (含 videoio + FFmpeg).
# 通过 adb pull 拉取头文件和 .so 到 3rdparty/opencv_board/,
# 交叉编译时动态链接, 运行时从板子系统路径加载.
find_package(OpenCV QUIET)
if(OpenCV_FOUND AND TARGET opencv_videoio)
    message(STATUS "找到 OpenCV ${OpenCV_VERSION} (含 videoio), 将构建视频检测程序 bishe_rknn_video")

    add_executable(bishe_rknn_video
        src/main_video.cc
        src/rknn_detector.cc
        src/postprocess.cc
    )

    target_include_directories(bishe_rknn_video PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LIBRKNNRT_INCLUDES}
        ${RKNN_MODEL_ZOO_ROOT}/utils
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(bishe_rknn_video
        fileutils
        imageutils
        ${LIBRKNNRT}
        ${OpenCV_LIBS}
        dl
    )

    # 交叉编译时, 板子的 OpenCV .so 依赖了大量系统库
    # (FFmpeg, GStreamer, tbb, lapack 等), 这些在编译机上不存在.
    # --allow-shlib-undefined 让链接器跳过 .so 中未解析的符号,
    # 运行时在板子上由动态链接器解析.
    target_link_options(bishe_rknn_video PRIVATE
        "LINKER:--allow-shlib-undefined"
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        target_link_libraries(bishe_rknn_video Threads::Threads)
    endif()

    install(TARGETS bishe_rknn_video DESTINATION .)
elseif(OpenCV_FOUND)
    message(WARNING "找到 OpenCV ${OpenCV_VERSION}, 但缺少 videoio 模块. "
                    "跳过 bishe_rknn_video 的构建.\n"
                    "  请用 adb pull 拉取板子的完整 OpenCV 到 3rdparty/opencv_board/")
else()
    message(WARNING "未找到 OpenCV, 跳过视频检测程序 bishe_rknn_video 的构建.\n"
                    "  请用 adb pull 拉取板子的完整 OpenCV 到 3rdparty/opencv_board/")
endif()