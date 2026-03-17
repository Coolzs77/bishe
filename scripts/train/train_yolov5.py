#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5红外目标检测训练脚本
使用官方YOLOv5训练流程
"""

import sys
import subprocess
import argparse
from pathlib import Path
import torch
import yaml

# ========== 自动定位项目根目录 ==========
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent




def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLOv5红外目标检测训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python train_yolov5.py --epochs 100 --batch-size 16 --name exp1
  python train_yolov5.py --img-size 416 --batch-size 32 --cache
        '''
    )

    # 基础配置
    parser.add_argument('--weights', type=str, default='yolov5s.pt',
                        help='预训练权重路径')
    parser.add_argument('--data', type=str, default='data/processed/flir/dataset.yaml',
                        help='数据集配置文件')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图像大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率')

    # 设备配置
    parser.add_argument('--device', type=str, default='0',
                        help='训练设备（GPU ID或cpu）')
    parser.add_argument('--workers', type=int, default=3,
                        help='数据加载线程数')

    # 优化选项
    parser.add_argument('--cache', action='store_true',
                        help='缓存图像到RAM（加速训练）')
    parser.add_argument('--cache-type', type=str, default='disk',
                        choices=['ram', 'disk'],
                        help='缓存类型')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')

    # 实验配置
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--project', type=str, default='outputs',
                        help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的权重路径')

    # 高级选项
    parser.add_argument('--hyp', type=str, default=None,
                        help='超参数配置文件')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='优化器类型')
    parser.add_argument('--cos-lr', action='store_true',
                        help='使用余弦学习率调度')
    parser.add_argument('--freeze', type=int, default=0,
                        help='冻结层数')

    return parser.parse_args()


def clone_yolov5():
    """自动克隆YOLOv5仓库"""
    print('\n正在克隆YOLOv5仓库...')
    print('-' * 60)

    yolov5_dir = PROJECT_ROOT / 'yolov5'

    try:
        # 克隆仓库
        subprocess.run(
            ['git', 'clone', 'https://github.com/ultralytics/yolov5.git', str(yolov5_dir)],
            check=True,
            cwd=str(PROJECT_ROOT),
            capture_output=False
        )
        print('✓ YOLOv5克隆成功')

        # 安装依赖
        print('\n安装依赖...')
        requirements_file = yolov5_dir / 'requirements.txt'
        if requirements_file.exists():
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file), '-q'],
                check=True
            )
            print('✓ 依赖安装成功')

        print('-' * 60)
        return True

    except subprocess.CalledProcessError as e:
        print(f'✗ 克隆失败: {e}')
        print('\n请手动克隆:')
        print('  git clone https://github.com/ultralytics/yolov5.git')
        print('-' * 60)
        return False
    except FileNotFoundError:
        print('✗ 未找到git命令')
        print('\n请先安装git，或手动克隆:')
        print('  git clone https://github.com/ultralytics/yolov5.git')
        print('-' * 60)
        return False


def check_environment():
    """检查训练环境"""
    print('\n[环境检查]')
    print('-' * 60)

    # 检查PyTorch
    print(f'PyTorch版本: {torch.__version__}')

    # 检查CUDA
    if torch.cuda.is_available():
        print(f'CUDA版本: {torch.version.cuda}')
        print(f'GPU数量: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f'  GPU {i}: {gpu_name} ({gpu_mem:.1f}GB)')
    else:
        print('CUDA: 不可用（将使用CPU训练）')

    # 检查YOLOv5
    yolov5_dir = PROJECT_ROOT / 'yolov5'
    if yolov5_dir.exists():
        print(f'YOLOv5目录: ✓ {yolov5_dir}')
    else:
        print(f'YOLOv5目录: ✗ 未找到')
        print('-' * 60)

        # 自动克隆
        if not clone_yolov5():
            return False

        print('\n[环境检查]')
        print('-' * 60)
        print(f'YOLOv5目录: ✓ {yolov5_dir}')

    print('-' * 60)
    return True


def check_data_config(data_config_path):
    """检查数据集配置"""
    print('\n[数据集检查]')
    print('-' * 60)

    config_path = PROJECT_ROOT / data_config_path

    if not config_path.exists():
        print(f'✗ 配置文件不存在: {config_path}')
        return False

    print(f'配置文件: ✓ {config_path}')

    # 读取配置
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 检查路径
    data_root = Path(config.get('path', ''))
    train_path = data_root / config.get('train', '')
    val_path = data_root / config.get('val', '')

    # 统计图像数量
    train_count = len(list(train_path.glob('*.jpg'))) if train_path.exists() else 0
    val_count = len(list(val_path.glob('*.jpg'))) if val_path.exists() else 0

    print(f'训练集: {train_path}')
    print(f'  图像数量: {train_count}')
    print(f'验证集: {val_path}')
    print(f'  图像数量: {val_count}')

    print(f'类别数量: {config.get("nc", 0)}')
    print(f'类别名称: {config.get("names", [])}')

    print('-' * 60)

    if train_count == 0:
        print('⚠ 警告: 未找到训练图像')
        return False

    return True


def check_weights(weights_path):
    """检查预训练权重"""
    print('\n[权重检查]')
    print('-' * 60)

    weights = PROJECT_ROOT / weights_path

    if weights.exists():
        size_mb = weights.stat().st_size / 1024 / 1024
        print(f'预训练权重: ✓ {weights}')
        print(f'文件大小: {size_mb:.1f} MB')
    else:
        print(f'预训练权重: ✗ 未找到 {weights}')
        print('\n正在下载预训练权重...')

        try:
            import urllib.request
            url = f'https://github.com/ultralytics/yolov5/releases/download/v7.0/{weights_path}'
            print(f'下载地址: {url}')

            urllib.request.urlretrieve(url, weights,
                                       reporthook=lambda a, b, c: print(f'\r下载进度: {a * b / c * 100:.1f}%', end=''))
            print('\n✓ 权重下载成功')
        except Exception as e:
            print(f'\n✗ 下载失败: {e}')
            print('\n请手动下载:')
            print(f'  https://github.com/ultralytics/yolov5/releases/download/v7.0/{weights_path}')
            print(f'  保存到: {weights}')
            return False

    print('-' * 60)
    return True


def print_training_config(args):
    """打印训练配置"""
    print('\n' + '=' * 60)
    print('YOLOv5 红外目标检测训练')
    print('=' * 60)
    print(f'实验名称: {args.name}')
    print(f'训练轮数: {args.epochs}')
    print(f'批量大小: {args.batch_size}')
    print(f'图像大小: {args.img_size}')
    print(f'学习率: {args.lr}')
    print(f'优化器: {args.optimizer}')
    print(f'设备: {args.device}')
    print(f'数据加载线程: {args.workers}')
    print('默认类别策略: 仅 person/car（自行车标签已删除）')
    if args.cache:
        print(f'图像缓存: ✓ ({args.cache_type})')
    else:
        print(f'图像缓存: ✗')
    print(f'早停耐心: {args.patience}')
    print(f'余弦学习率: {"✓" if args.cos_lr else "✗"}')
    if args.freeze > 0:
        print(f'冻结层数: {args.freeze}')
    print('=' * 60 + '\n')


def build_train_command(args):
    """构建训练命令"""

    # 基础命令
    cmd = [
        sys.executable,
        'train.py',
        '--img', str(args.img_size),
        '--batch', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--data', str((PROJECT_ROOT / args.data).absolute()),
        '--weights', str((PROJECT_ROOT / args.weights).absolute()),
        '--project', str((PROJECT_ROOT / args.project).absolute()),
        '--name', args.name,
        '--device', args.device,
        '--workers', str(args.workers),
    ]

    # 缓存选项
    if args.cache:
        cmd.extend(['--cache', args.cache_type])

    # 早停
    cmd.extend(['--patience', str(args.patience)])

    # 优化器
    cmd.extend(['--optimizer', args.optimizer])

    # 余弦学习率
    if args.cos_lr:
        cmd.append('--cos-lr')

    # 冻结层
    if args.freeze > 0:
        cmd.extend(['--freeze', str(args.freeze)])

    # 超参数配置
    if args.hyp:
        cmd.extend(['--hyp', str((PROJECT_ROOT / args.hyp).absolute())])

    # 恢复训练
    if args.resume:
        cmd = [sys.executable, 'train.py', '--resume', str((PROJECT_ROOT / args.resume).absolute())]

    return cmd


def estimate_training_time(args, train_count):
    """估算训练时间"""
    print('\n[训练时间估算]')
    print('-' * 60)

    # 根据图像大小和batch size估算
    batches_per_epoch = (train_count + args.batch_size - 1) // args.batch_size

    # 估算每batch时间（经验值）
    if args.img_size <= 416:
        time_per_batch = 0.8
    elif args.img_size <= 640:
        time_per_batch = 1.5
    else:
        time_per_batch = 2.5

    # 如果使用缓存，加速
    if args.cache:
        time_per_batch *= 0.6

    epoch_time = batches_per_epoch * time_per_batch
    total_time = epoch_time * args.epochs

    print(f'每轮批次数: {batches_per_epoch}')
    print(f'预计每批次: {time_per_batch:.1f}秒')
    print(f'预计每轮: {epoch_time / 60:.1f}分钟')
    print(f'预计总时长: {total_time / 3600:.1f}小时')
    print('-' * 60)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 打印配置
    print_training_config(args)

    # 检查环境
    if not check_environment():
        print('\n✗ 环境检查失败')
        return

    # 检查数据集
    if not check_data_config(args.data):
        print('\n✗ 数据集检查失败')
        return

    # 检查权重
    if not args.resume and not check_weights(args.weights):
        print('\n✗ 权重检查失败')
        return

    # 估算训练时间
    data_config_path = PROJECT_ROOT / args.data
    with open(data_config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_path = Path(config.get('path', '')) / config.get('train', '')
    train_count = len(list(train_path.glob('*.jpg')))
    estimate_training_time(args, train_count)

    # 构建命令
    cmd = build_train_command(args)

    # 打印命令
    print('\n[执行命令]')
    print('-' * 60)
    print(' '.join(cmd))
    print('-' * 60 + '\n')

    # 确认开始
    print('🚀 准备开始训练...')
    print('提示: 按 Ctrl+C 可以中断训练\n')

    # 执行训练
    yolov5_dir = PROJECT_ROOT / 'yolov5'

    try:
        result = subprocess.run(
            cmd,
            cwd=str(yolov5_dir),
            check=True
        )

        # 训练完成
        print('\n' + '=' * 60)
        print('✅ 训练完成!')
        print('=' * 60)

        output_dir = PROJECT_ROOT / args.project / args.name
        print(f'\n结果保存在: {output_dir}')
        print(f'  - 最佳模型: {output_dir}/weights/best.pt')
        print(f'  - 最终模型: {output_dir}/weights/last.pt')
        print(f'  - 训练曲线: {output_dir}/results.png')
        print(f'  - 混淆矩阵: {output_dir}/confusion_matrix.png')

        print('\n查看训练曲线:')
        print(f'  tensorboard --logdir {output_dir}')

        print('\n恢复训练:')
        print(f'  python {__file__} --resume {output_dir}/weights/last.pt')

        print('\n' + '=' * 60 + '\n')

    except subprocess.CalledProcessError as e:
        print('\n' + '=' * 60)
        print('✗ 训练失败')
        print('=' * 60)
        print(f'错误码: {e.returncode}')
        print('\n请检查上方的错误信息')

    except KeyboardInterrupt:
        print('\n\n' + '=' * 60)
        print('⚠ 训练已中断')
        print('=' * 60)
        print(f'\n可以通过以下命令恢复训练:')
        output_dir = PROJECT_ROOT / args.project / args.name
        print(f'  python {__file__} --resume {output_dir}/weights/last.pt')
        print('\n' + '=' * 60 + '\n')


if __name__ == '__main__':
    main()