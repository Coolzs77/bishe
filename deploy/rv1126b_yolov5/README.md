# RV1126B 红外目标检测部署指南

本目录是红外 YOLOv5 检测模型在 RV1126B 开发板上的完整部署代码，支持红外图片检测和红外视频检测。模型输入为 640×640×3 的红外热成像图片（虽然红外图像看起来是灰度的，但 JPEG 加载后自动成为三通道），输出为 3 个分支的原始卷积特征图（P3/P4/P5），板端 C++ 代码完成 sigmoid + grid/anchor 解码。2 类：person、car。

**部署模型**：`ablation_exp07_eiou`（基线 + EIoU），候选：`ablation_exp09_ghost_eiou`（Ghost + EIoU 轻量化）。

---

## 目录结构

```text
deploy/rv1126b_yolov5/
├── CMakeLists.txt              # CMake 构建配置
├── build_rv1126b.sh            # 一键交叉编译脚本 (自动拉取板子 RGA + OpenCV)
├── calibration_dataset.txt     # 量化校准集列表 (prepare_deploy.py 生成)
├── README.md                   # 本文档
├── 3rdparty/
│   ├── opencv_board/           # 从板子 adb pull 的 OpenCV 4.6.0 (含 videoio)
│   └── rga_board/              # 从板子 adb pull 的 librga.so (匹配内核驱动)
├── model/
│   ├── infrared_labels.txt     # 标签文件 (person, car)
│   └── best_eiou.onnx          # ONNX 模型 (3-branch 输出, prepare_deploy.py 生成)
├── python/
│   └── convert_yolov5_to_rknn.py  # ONNX → RKNN 转换脚本
├── testdata/                   # 测试数据 (prepare_deploy.py 生成)
│   ├── test_00.jpg ~ test_04.jpg  # FLIR 红外测试图片
│   └── ZAtDSNuZZjkZFvMAo_seq006.mp4  # 红外测试视频
└── src/
    ├── main.cc                 # 板端图片检测程序
    ├── main_video.cc           # 板端视频检测 / 摄像头实时推理
    ├── postprocess.cc          # 后处理 (3-branch 解码 + NMS)
    ├── postprocess.hpp         # 后处理头文件 (模型常量 + anchor)
    ├── rknn_detector.cc        # RKNN 推理封装
    └── rknn_detector.hpp       # RKNN 推理头文件
```

---

## 部署操作流程

整个部署分 3 步，分别在 3 台机器上完成：

### 第一步：Windows — 导出 ONNX 模型

在你的 Windows 训练机上完成。

#### 1.1 一键准备（推荐）

使用部署准备脚本自动导出 ONNX + 准备测试数据：

```powershell
cd D:\pythonPro\bishe

# 导出基线+eiou 模型（主力）
python scripts/deploy/prepare_deploy.py --exp eiou

# 或导出 ghost+eiou 模型（轻量化候选, 注意用 --imgsz 640）
python scripts/deploy/prepare_deploy.py --exp ghost_eiou --imgsz 640
```

脚本会自动完成：

- 从 `outputs/ablation_study/ablation_exp07_eiou/weights/best.pt` 导出 **3-branch ONNX**（3 个原始卷积输出，INT8 量化精度远好于单输出格式）
- 从 FLIR 验证集随机选取 150 张图片生成量化校准列表
- 选取 5 张测试图片 + 2 个测试视频到 `deploy/rv1126b_yolov5/testdata/`

#### 1.2 手动导出（可选）

```powershell
cd D:\pythonPro\bishe
python yolov5/export.py ^
  --weights outputs/ablation_study/ablation_exp07_eiou/weights/best.pt ^
  --include onnx --imgsz 640 --batch-size 1 --simplify --device cpu
```

导出完成后得到 `best.onnx`，确认有 3 个输出节点（P3/P4/P5 raw conv，形状分别为 `[1,21,80,80]`、`[1,21,40,40]`、`[1,21,20,20]`）。

> **为什么用 3-branch 而非单输出？** 单输出 `[1,25200,7]` 混合了坐标 (0~640) 和置信度 (0~1)，INT8 量化步长 ~5.7，置信度全部被量化为 0，导致检测不到目标。3-branch 输出每个张量值域更均匀，量化精度高。

#### 1.3 拷贝文件到 Ubuntu

在 Ubuntu 的家目录下建好工作目录，然后把文件拷贝过去：

```bash
# Ubuntu 端先建好目录结构
mkdir -p /home/coolzs77/bishe
```

把以下文件从 Windows 拷贝到 Ubuntu 的 `/home/coolzs77/bishe/` 下（scp / 共享文件夹 / U盘 均可）：

- `deploy/rv1126b_yolov5/` → `/home/coolzs77/bishe/deploy/rv1126b_yolov5/`
- `data/processed/flir/images/val/` → `/home/coolzs77/bishe/val/`（直接放在 bishe 根目录下）
- `rknn_model_zoo/` → `/home/coolzs77/bishe/rknn_model_zoo/`

---

### 第二步：Ubuntu — 转换模型 + 交叉编译

在 Ubuntu 中完成，工作目录为 `/home/coolzs77/bishe/`。

#### 2.1 安装 RKNN-Toolkit2

如果还没装，先安装：

```bash
# 创建 Python 3.8 虚拟环境
conda create -n rknn python=3.8 -y
conda activate rknn

# 安装 rknn-toolkit2 (根据你下载的版本)
pip install rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

#### 2.2 ONNX → RKNN 模型转换

```bash
cd /home/coolzs77/bishe/deploy/rv1126b_yolov5/python

# 基线+eiou 模型 (主力)
python convert_yolov5_to_rknn.py \
  --onnx ../model/best_eiou.onnx \
  --dataset ../calibration_dataset.txt \  --val-dir /home/coolzs77/bishe/val \  --output ../model/best_eiou.rknn \
  --target rv1126b \
  --quant i8

# 或 ghost+eiou 模型 (轻量化候选)
# python convert_yolov5_to_rknn.py \
#   --onnx ../model/best_ghost_eiou.onnx \
#   --dataset ../calibration_dataset.txt \
#   --output ../model/best_ghost_eiou.rknn \
#   --target rv1126b --quant i8
```

> **注意**：`calibration_dataset.txt` 中的图片路径是 Windows 格式。不需要手动执行 sed 替换，直接给转换脚本加上 `--val-dir` 参数，脚本会自动把每行的文件名拼到该目录下。

参数说明：

| 参数        | 含义                                    |
| ----------- | --------------------------------------- |
| `--onnx`    | 第一步导出的 ONNX 模型路径              |
| `--dataset` | 红外校准图片列表文件                    |
| `--output`  | 输出 RKNN 模型路径                      |
| `--target`  | 目标芯片，填 `rv1126b`                  |
| `--quant`   | `i8` = INT8 量化（推荐），`fp` = 浮点   |

转换成功后会在 `model/` 目录下生成 `.rknn` 文件。

> 如果 OpenCV 可用（RV1126B SDK 通常自带），还会构建 `bishe_rknn_video` 视频检测程序。

#### 2.3 安装交叉编译工具链

```bash
sudo apt update
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu cmake
```

#### 2.4 拉取板子依赖库 + 交叉编译

`build_rv1126b.sh` 会自动检测并从板子拉取 RGA 库（需要 adb 连接板子）：

```bash
cd /home/coolzs77/bishe/deploy/rv1126b_yolov5

# 确保板子已通过 USB 连接
adb devices

# 一键编译（自动拉取 librga.so）
bash build_rv1126b.sh
```

> **RGA 说明**: rknn_model_zoo 自带的 librga v1.10.1 与板子内核 RGA 驱动 v4.x 不兼容，会报 `rga2 get info failed` 并回退 CPU。脚本会自动从板子拉取匹配版本的 `librga.so`，让 RGA 硬件加速正常工作。

如果 adb 不可用，可手动拉取：

```bash
mkdir -p 3rdparty/rga_board
adb pull /usr/lib/aarch64-linux-gnu/librga.so 3rdparty/rga_board/librga.so
bash build_rv1126b.sh
```

如果你的编译器路径不是默认的，先导出：

```bash
export GCC_COMPILER=/path/to/aarch64-linux-gnu
bash build_rv1126b.sh
```

编译成功后，输出目录：

```text
deploy/rv1126b_yolov5/install/rv1126b_linux_aarch64/bishe_rknn_yolov5/
├── bishe_rknn_detect          # 图片检测可执行文件
├── bishe_rknn_video           # 视频检测可执行文件 (需要 OpenCV)
├── lib/                       # RKNN 运行时库
└── model/
    └── infrared_labels.txt    # 标签文件
```

#### 2.5 把模型和测试数据放进安装目录

```bash
cd /home/coolzs77/bishe/deploy/rv1126b_yolov5

# 复制 RKNN 模型
cp model/best_eiou.rknn install/rv1126b_linux_aarch64/bishe_rknn_yolov5/model/

# 复制测试图片和视频
cp testdata/* install/rv1126b_linux_aarch64/bishe_rknn_yolov5/model/
```

---

### 第三步：RV1126B 开发板 — 推送和运行

#### 3.1 连接开发板

用 USB 线连接 RV1126B 开发板到 Ubuntu 主机（或 Windows，均可），确认 adb 可用：

```bash
adb devices
# 应该看到你的设备
```

#### 3.2 推送文件到板子

```bash
cd /home/coolzs77/bishe/deploy/rv1126b_yolov5
adb push install/rv1126b_linux_aarch64/bishe_rknn_yolov5 /userdata/
```

#### 3.3 登录板子并运行

```bash
adb shell
cd /userdata/bishe_rknn_yolov5
export LD_LIBRARY_PATH=./lib

# ============ 红外图片检测 ============
./bishe_rknn_detect model/best_eiou.rknn model/test_00.jpg model/infrared_labels.txt out.png
# 可以依次测试多张: test_01.jpg, test_02.jpg ...

# ============ 红外视频检测 ============
./bishe_rknn_video model/best_eiou.rknn model/ZAtDSNuZZjkZFvMAo_seq006.mp4 \
    model/infrared_labels.txt out_video.mp4

# 自定义阈值 (置信度=0.3, NMS=0.5)
./bishe_rknn_video model/best_eiou.rknn model/ZAtDSNuZZjkZFvMAo_seq006.mp4 \
    model/infrared_labels.txt out_video.mp4 0.3 0.5

# ============ 红外摄像头实时推理 (后期接上红外摄像头时) ============
# ./bishe_rknn_video model/best_eiou.rknn /dev/video0 \
#     model/infrared_labels.txt camera_out.mp4
# 或
# ./bishe_rknn_video model/best_eiou.rknn camera:0 \
#     model/infrared_labels.txt camera_out.mp4
```

图片检测输出示例：

```text
=== 红外目标检测 ===
  模型:   model/best_eiou.rknn
  图片:   model/test_00.jpg
  标签:   model/infrared_labels.txt
  输出:   out.png
  conf:   0.25
  nms:    0.45

红外图片已加载: 640x512, format=0

检测到 3 个目标:
  [0] person @ (120,80)-(200,300) 87.3%
  [1] person @ (350,100)-(420,310) 72.1%
  [2] car @ (50,250)-(280,400) 65.8%

结果已保存: out.png
```

视频检测输出示例：

```text
=== 红外视频检测 ===
  模型:   model/best_eiou.rknn
  输入:   model/ZAtDSNuZZjkZFvMAo_seq006.mp4
  ...

视频信息: 640x512, 30.0 FPS, 共 300 帧

  进度: 100/300 帧 (33%), 平均推理: 28.5 ms, NPU FPS: 35.1
  进度: 200/300 帧 (67%), 平均推理: 27.8 ms, NPU FPS: 36.0
  进度: 300/300 帧 (100%), 平均推理: 27.6 ms, NPU FPS: 36.2

=== 推理完成 ===
  总帧数:     300
  总检测数:   856
  平均推理:   27.6 ms/帧
  NPU FPS:    36.2
  端到端 FPS: 24.1 (含视频读写)
  输出视频:   out_video.mp4
```

#### 3.4 取回结果

```bash
# 在 Ubuntu 或 Windows 主机上执行

# 取回图片检测结果
adb pull /userdata/bishe_rknn_yolov5/out.png ./

# 取回视频检测结果
adb pull /userdata/bishe_rknn_yolov5/out_video.mp4 ./
```

用图片查看器打开 `out.png`，用播放器打开 `out_video.mp4`，检查检测框是否正确。

---

## 红外图像处理说明

### 为什么模型输入是 3 通道？

训练时使用 OpenCV 的 `cv2.imread()` 读取 FLIR 红外 JPEG 图片，OpenCV 默认输出 BGR 三通道（灰度 JPEG 的三通道值相同）。所以 YOLOv5 模型的输入层是 640×640×3。

### 板端如何处理红外图像？

1. **JPEG/PNG 红外图片**：`read_image()` 自动解码为 RGB888（三通道），和训练一致，直接送入模型。
2. **原始单通道灰度帧**（如 V4L2 摄像头）：推理函数中的 `ensure_rgb888()` 会自动复制灰度值到三个通道。

### 训练时的红外特殊设置

训练超参已关闭颜色增强（红外无色彩信息）：

```yaml
hsv_h: 0.0    # 色调增强 - 关闭
hsv_s: 0.0    # 饱和度增强 - 关闭
hsv_v: 0.3    # 亮度增强 - 保留 (模拟红外亮度变化)
```

---

## 模型要求

本部署代码针对以下消融实验模型：

| 模型 | 配置 | 描述 |
| --- | --- | --- |
| **best_eiou** (主力) | yolov5s_base.yaml | 基线 + EIoU Loss，640×640 训练 |
| **best_ghost_eiou** (候选) | yolov5s_lightweight.yaml | C3Ghost + EIoU Loss，704×704 训练 → 640 导出 |

模型要求：

1. **输入**：640×640×3 红外图像（灰度三通道）
2. **输出**：3 个分支 (P3/P4/P5 原始卷积输出)，形状分别为 `[1,21,80,80]`、`[1,21,40,40]`、`[1,21,20,20]`
3. **类别**：2 类（person=0, car=1）

> **注意**：ghost_eiou 训练时使用 704×704，但导出 ONNX 时用 `--imgsz 640` 统一到 640，部署代码无需修改。
>
> **为什么用 3-branch 输出？** 单输出 `[1,25200,7]` 混合了坐标值 (0~640) 和置信度 (0~1)，INT8 量化步长过大会把所有置信度量化为 0，导致检测 0 个目标。3-branch 输出每个张量值域均匀，INT8 量化精度高。板端 C++ 代码自动完成 sigmoid + grid/anchor 解码。
>
> **两种输出格式均支持**：推理代码会自动检测 RKNN 模型的输出数量（1 个或 3 个），无需手动配置。但强烈推荐 3-branch。

---

## 常见问题

### Q: 量化后检测 0 个目标？

A: 最可能的原因是使用了 **单输出 ONNX** (`[1,25200,7]`)。该格式混合了坐标 (0~640) 和置信度 (0~1)，INT8 量化步长 ~5.7，置信度全部量化为 0。**必须使用 3-branch ONNX 输出格式**（用 `prepare_deploy.py` 导出的即是 3-branch）。如需确认，可用 `--quant fp` 测试浮点精度。

### Q: RGA 报错 "rga2 get info failed"？

A: rknn_model_zoo 自带的 librga v1.10.1 与板子内核 RGA 驱动版本不兼容。解决方案：从板子 adb pull 拉取匹配的 librga.so 到 `3rdparty/rga_board/`。`build_rv1126b.sh` 会自动拉取。

### Q: 板子上运行报 `rknn_init failed`？

A: 检查 RKNN 模型是否是为 `rv1126b` 平台编译的，以及 `lib/` 目录下的 `librknnrt.so` 版本是否匹配。

### Q: 检测不到目标？

A: 试着降低置信度阈值（如 0.1），检查输出是否有候选框。如果完全没有，可能是 ONNX 导出或量化出了问题。

### Q: 视频推理速度慢？

A: 纯 NPU 推理通常 30+ FPS，但端到端 FPS（含视频解码/编码）会更低。排查：

1. 确认 OpenCV 是带硬件编解码支持编译的（RV1126B SDK 通常自带）
2. 用较短的视频（如 300 帧）测试排除 I/O 瓶颈
3. 检查 `avg inference` 时间，如果 <35ms 说明 NPU 端正常

### Q: 视频推理时 `VideoCapture` 打不开？

A: 确认视频文件已正确推送到板子，且路径无误。部分 MP4 编码可能不被板端 OpenCV 支持，可先在 PC 上用 ffmpeg 转码为 H.264 baseline profile。

### Q: ghost_eiou 模型部署后精度不行？

A: ghost_eiou 训练时使用了 704×704 输入尺寸，但部署统一用 640×640。这可能导致轻微精度损失，属正常现象。若需最佳精度可修改 `postprocess.hpp` 中的 `MODEL_INPUT_SIZE` 为 704，并重新导出/转换。
