#!/bin/bash
# ========================================================================
#  build_rv1126b.sh — 一键交叉编译脚本 (Ubuntu / WSL 上运行)
# ========================================================================
#
#  用途:  在 Ubuntu 上交叉编译出 aarch64 的板端可执行文件,
#         然后把安装目录拷贝到 RV1126B 板子上运行.
#
#  前置条件:
#    1. 已安装 aarch64-linux-gnu-gcc / g++ 交叉编译工具链
#       Ubuntu: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
#    2. rknn_model_zoo 已放在 ../../rknn_model_zoo 位置
#
#  输出:
#    install/rv1126b_linux_aarch64/bishe_rknn_yolov5/
#      ├─ bishe_rknn_detect        可执行文件
#      ├─ lib/                     动态库 (librknnrt.so 等)
#      └─ model/
#           ├─ infrared_labels.txt   标签文件
#           └─ *.rknn               RKNN 模型
# ========================================================================

# 遇到任何命令出错立即退出, 避免在错误状态下继续执行.
set -e

# ---------- 路径变量 ----------
# ROOT_DIR: 本脚本所在的目录 (deploy/rv1126b_yolov5/).
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

# BUILD_DIR: CMake 构建产物目录 (中间文件, 不会被提交).
BUILD_DIR="${ROOT_DIR}/build"

# INSTALL_DIR: 最终安装目录, 拷贝到板子运行的就是这个文件夹.
INSTALL_DIR="${ROOT_DIR}/install/rv1126b_linux_aarch64/bishe_rknn_yolov5"

# ---------- 编译器配置 ----------
# GCC_PREFIX: 交叉编译器前缀, 默认 aarch64-linux-gnu.
# 你也可以通过环境变量覆盖, 例如:
#   GCC_COMPILER=aarch64-none-linux-gnu bash build_rv1126b.sh
GCC_PREFIX="${GCC_COMPILER:-aarch64-linux-gnu}"

# CC_BIN / CXX_BIN: 实际使用的 C/C++ 编译器路径.
# 优先取环境变量 CC / CXX, 这样你可以灵活指定.
CC_BIN="${CC:-${GCC_PREFIX}-gcc}"
CXX_BIN="${CXX:-${GCC_PREFIX}-g++}"

# ---------- 校验编译器是否存在 ----------
# 找不到编译器时直接报错退出, 避免后续 cmake configure 报难懂的错.
if ! command -v "${CC_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] C 编译器未找到: ${CC_BIN}"
    echo "  请先安装: sudo apt install gcc-aarch64-linux-gnu"
    exit 1
fi

if ! command -v "${CXX_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] C++ 编译器未找到: ${CXX_BIN}"
    echo "  请先安装: sudo apt install g++-aarch64-linux-gnu"
    exit 1
fi

# ---------- CMake 配置 (configure) ----------
# -S: 源码目录 (本目录, 里面有 CMakeLists.txt)
# -B: 构建目录 (build/, 不污染源码)
#
# 交叉编译关键参数:
#   CMAKE_SYSTEM_NAME=Linux       告诉 CMake 目标系统是 Linux
#   CMAKE_SYSTEM_PROCESSOR=aarch64  目标架构 ARM 64 位
#   CMAKE_C_COMPILER / CMAKE_CXX_COMPILER  指定交叉编译器
#   CMAKE_BUILD_TYPE=Release      开启优化, 生产环境必备
#   TARGET_SOC=rv1126b            告诉 rknn 3rdparty 选择对应的库
#   CMAKE_INSTALL_PREFIX          安装路径
#   OpenCV_DIR                    指向板子上拉取的 OpenCV 4.6.0

# ---------- OpenCV 路径 ----------
# 使用从 RV1126B 板子上 adb pull 拉取的 OpenCV 4.6.0 (含 videoio 动态库).
# 如果你的板子有不同版本的 OpenCV, 重新执行拉取步骤.
RKNN_MODEL_ZOO_ROOT="${ROOT_DIR}/../../rknn_model_zoo"
OPENCV_BOARD="${ROOT_DIR}/3rdparty/opencv_board"

if [ -f "${OPENCV_BOARD}/OpenCVConfig.cmake" ]; then
    OPENCV_DIR="${OPENCV_BOARD}"
    echo "[INFO] 使用板子 OpenCV 4.6.0 (含 videoio): ${OPENCV_DIR}"

    # 创建 .so 符号链接 (链接器需要不带版本号的 .so).
    # 如果已经存在则跳过.
    for f in "${OPENCV_BOARD}"/lib/libopencv_*.so.4.6.0; do
        base=$(basename "$f" .so.4.6.0)
        link="${OPENCV_BOARD}/lib/${base}.so"
        if [ ! -e "${link}" ]; then
            ln -sf "$(basename "$f")" "${link}"
            echo "  创建符号链接: $(basename "${link}") -> $(basename "$f")"
        fi
    done
else
    # 回退到 rknn_model_zoo 精简版 (不含 videoio, 只能编译图片检测).
    OPENCV_DIR="${RKNN_MODEL_ZOO_ROOT}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV"
    echo "[WARN] 未找到板子 OpenCV, 回退到 rknn_model_zoo 精简版."
    echo "       视频检测程序 bishe_rknn_video 将无法编译."
    echo "       请参考 README.md 用 adb pull 拉取板子的 OpenCV."
fi

# ---------- RGA 库 ----------
# rknn_model_zoo 自带的 librga.a (v1.10.1) 与板子内核 RGA 驱动 (v4.x) 版本不兼容,
# 运行时报 "rga2 get info failed" 并回退 CPU.
# 解决方案: 从板子 adb pull 拉取匹配版本的 librga.so, 动态链接.
RGA_BOARD="${ROOT_DIR}/3rdparty/rga_board"

if [ ! -f "${RGA_BOARD}/librga.so" ]; then
    echo "[INFO] 板子 RGA 库不存在, 尝试从板子 adb pull 拉取..."
    if command -v adb >/dev/null 2>&1 && adb devices 2>/dev/null | grep -q "device$"; then
        mkdir -p "${RGA_BOARD}"
        # 板子上的 librga.so 可能在不同路径, 依次尝试
        RGA_PULLED=0
        for rga_path in \
            /usr/lib/aarch64-linux-gnu/librga.so \
            /usr/lib/librga.so \
            /usr/lib64/librga.so \
            /vendor/lib64/librga.so; do
            if adb shell "test -f ${rga_path}" 2>/dev/null; then
                adb pull "${rga_path}" "${RGA_BOARD}/librga.so"
                RGA_PULLED=1
                echo "[INFO] 已从板子拉取 RGA 库: ${rga_path}"
                break
            fi
        done
        if [ ${RGA_PULLED} -eq 0 ]; then
            echo "[WARN] 板子上未找到 librga.so, RGA 加速不可用."
            echo "       可在板子上执行: find / -name 'librga*' 2>/dev/null"
        fi
    else
        echo "[WARN] adb 不可用或板子未连接, 跳过 RGA 库拉取."
        echo "       请手动: adb pull /usr/lib/aarch64-linux-gnu/librga.so ${RGA_BOARD}/"
    fi
fi

if [ -f "${RGA_BOARD}/librga.so" ]; then
    echo "[INFO] 使用板子 RGA 库: ${RGA_BOARD}/librga.so"
else
    echo "[WARN] RGA 库缺失, 图像预处理将回退 CPU (不影响 NPU 推理性能)."
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTARGET_SOC=rv1126b \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER="${CC_BIN}" \
    -DCMAKE_CXX_COMPILER="${CXX_BIN}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DOpenCV_DIR="${OPENCV_DIR}"

# ---------- 编译 + 安装 ----------
# --build:  执行编译
# --target install: 编译完成后自动复制到 INSTALL_DIR
# -j4:      并行 4 线程编译, 加快速度
cmake --build "${BUILD_DIR}" --target install -j4

# ---------- 完成提示 ----------
echo ""
echo "========================================"
echo "  编译完成!"
echo "  安装目录: ${INSTALL_DIR}"
echo ""
echo "  下一步:"
echo "    1. 把 ${INSTALL_DIR} 整个目录拷贝到板子"
echo "    2. 在板子上执行:"
echo "       cd bishe_rknn_yolov5"
echo "       ./bishe_rknn_detect model/xxx.rknn"
echo "========================================"
