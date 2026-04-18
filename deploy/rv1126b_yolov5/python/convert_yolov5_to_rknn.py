# -*- coding: utf-8 -*-
import argparse
import os
import tempfile
from pathlib import Path

from rknn.api import RKNN


# ========================================================================
#  红外 YOLOv5 模型 ONNX → RKNN 转换脚本
# ========================================================================
#
# 运行环境: Ubuntu 或 WSL (需要安装 rknn-toolkit2)
# 目标平台: RV1126B (NPU)
#
# 本脚本将训练好的红外 YOLOv5 ONNX 模型转换为 RV1126B 可运行的 RKNN 格式.
# 支持 INT8 量化和 FP 浮点两种模式.
#
# 关于红外图像量化校准:
#   INT8 量化需要提供校准数据集 (dataset.txt), 里面列出的图片应该是
#   你自己的红外热成像 JPEG 图片 (来自 FLIR 数据集的训练/验证集).
#   校准集应覆盖各种场景: 白天/夜间, 近景/远景, 行人/车辆等.
#   建议选取 50~100 张有代表性的红外图片. (注：为防止内存溢出，不建议超过100张)
#
# 关于 mean/std:
#   训练时 YOLOv5 的预处理是 pixel / 255.0, 这等价于:
#   mean=[0, 0, 0], std=[255, 255, 255]
#   红外灰度图虽然三通道值相同, 但 mean/std 设置不变.
# ========================================================================


# 把命令行里的三元组字符串解析成浮点数组.
# 例如: --mean 0,0,0  →  [0.0, 0.0, 0.0]
def parse_triplet(text):
    values = [float(item.strip()) for item in text.split(',') if item.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError('需要 3 个逗号分隔的值, 例如 0,0,0')
    return values


# 解析脚本参数.
def parse_args():
    parser = argparse.ArgumentParser(
        description='将红外 YOLOv5 ONNX 模型转换为 RV1126B RKNN 格式')
    parser.add_argument('--onnx', required=True,
                        help='输入 ONNX 模型路径')
    parser.add_argument('--dataset', required=False,
                        help='量化校准集文件 (每行一张红外图片路径)')
    parser.add_argument('--output', required=True,
                        help='输出 RKNN 模型路径')
    parser.add_argument('--target', default='rv1126b',
                        help='目标平台, 默认: rv1126b')
    parser.add_argument('--quant', choices=['i8', 'fp'], default='i8',
                        help='i8=INT8量化, fp=浮点 (默认: i8)')
    parser.add_argument('--opt-level', type=int, default=3, choices=[0, 1, 2, 3],
                        help='RKNN 优化等级 0~3 (默认: 3, 最大优化)')
    
    # 核心修改点：加入 kl_divergence 并设为默认值，完美适配 EIOU 模型，解决重叠框和内存爆炸问题
    parser.add_argument('--quant-algo', choices=['normal', 'mmse', 'kl_divergence'], default='kl_divergence',
                        help='量化算法: normal=默认, mmse=最小均方误差, kl_divergence=KL散度(强烈推荐EIOU使用) (默认: kl_divergence)')
    
    parser.add_argument('--mean', type=parse_triplet, default=[0.0, 0.0, 0.0],
                        help='均值, 默认: 0,0,0')
    parser.add_argument('--std', type=parse_triplet, default=[255.0, 255.0, 255.0],
                        help='标准差, 默认: 255,255,255')
    parser.add_argument('--val-dir', default=None,
                        help='Ubuntu 上校准图片所在目录 (用于自动修正 dataset 里的 Windows 路径, '
                             '例如 /home/coolzs77/bishe/val)')
    return parser.parse_args()


def fix_dataset_paths(dataset_path, val_dir):
    """把 dataset 文件里的 Windows 路径替换为 Linux 路径.

    只取每行路径的文件名部分, 拼上 val_dir, 写入临时文件后返回临时文件路径.
    调用者负责在使用后删除该临时文件.
    """
    val_dir = Path(val_dir).expanduser().resolve()
    fixed_lines = []
    skipped = 0
    with open(dataset_path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # 统一把反斜杠转为正斜杠, 再取文件名
            fname = Path(line.replace('\\', '/')).name
            fixed = val_dir / fname
            if not fixed.exists():
                skipped += 1
                continue
            fixed_lines.append(str(fixed))

    if not fixed_lines:
        raise FileNotFoundError(
            '--val-dir={} 下找不到任何 dataset 中列出的图片, '
            '请确认目录正确且图片已传输到 Ubuntu.'.format(val_dir))

    if skipped:
        print('[WARN] {} 张图片在 {} 中不存在, 已跳过'.format(skipped, val_dir))
    print('[INFO] dataset 路径已修正: {} 张图片 → {}'.format(len(fixed_lines), val_dir))

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                      delete=False, encoding='utf-8')
    for p in fixed_lines:
        tmp.write(p + '\n')
    tmp.close()
    return tmp.name


def main():
    args = parse_args()

    # 转成绝对路径, 避免在不同工作目录下执行时找不到文件.
    onnx_path = Path(args.onnx).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    dataset_path = None
    if args.dataset:
        dataset_path = Path(args.dataset).expanduser().resolve()

    # INT8 量化必须提供红外图片校准集; FP 模式可以不传.
    if not onnx_path.exists():
        raise FileNotFoundError('ONNX 文件不存在: {}'.format(onnx_path))
    if args.quant == 'i8' and (dataset_path is None or not dataset_path.exists()):
        raise FileNotFoundError('INT8 量化需要 --dataset 指定红外校准图片列表文件')

    # 如果提供了 --val-dir, 自动修正 dataset 里的 Windows 路径
    tmp_dataset = None
    if dataset_path and args.val_dir:
        tmp_dataset = fix_dataset_paths(dataset_path, args.val_dir)
        dataset_path = Path(tmp_dataset)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建 RKNN 工具对象.
    rknn = RKNN(verbose=False)

    print('--> 配置模型参数')
    # 设置预处理参数和目标平台.
    #
    # 红外 YOLOv5 训练时的预处理: pixel / 255.0
    # 等价于 mean=[0,0,0], std=[255,255,255].
    # 虽然红外图像三通道值相同, 但仍需按三通道设置.
    rknn.config(
        mean_values=[args.mean],
        std_values=[args.std],
        target_platform=args.target,
        optimization_level=args.opt_level,
        quantized_algorithm=args.quant_algo if args.quant == 'i8' else 'normal',
    )
    print('done')

    print('--> 加载 ONNX 模型')
    # 读入 ONNX 图结构, 此步仅在 Toolkit2 内存中操作.
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError('加载 ONNX 失败: {}'.format(ret))
    print('done')

    print('--> 构建 RKNN 模型 (转换 + 量化)')
    # build 阶段执行实际的模型转换和 INT8 量化.
    #
    # 如果选择 i8 量化, 会读取 dataset.txt 中列出的红外图片做量化校准.
    # 校准图片应该是你自己的 FLIR 红外热成像 JPEG (不是普通 RGB 照片).
    ret = rknn.build(
        do_quantization=(args.quant == 'i8'),
        dataset=str(dataset_path) if dataset_path else None,
    )
    if ret != 0:
        raise RuntimeError('构建失败: {}'.format(ret))
    print('done')

    print('--> 导出 RKNN 文件')
    # 导出最终的 .rknn 文件, 这就是要上传到 RV1126B 板子上的模型.
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise RuntimeError('导出失败: {}'.format(ret))
    print('done')

    # 释放 Toolkit2 资源.
    rknn.release()

    # 清理临时 dataset 文件
    if tmp_dataset and Path(tmp_dataset).exists():
        os.unlink(tmp_dataset)

    print('RKNN 模型已导出: {}'.format(output_path))


if __name__ == '__main__':
    main()