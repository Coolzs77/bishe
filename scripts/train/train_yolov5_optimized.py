#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5热红外训练脚本 - 支持优化损失函数

支持配置：
- 标准YOLOv5损失
- EIoU Loss优化
- Focal Loss优化
- 两者组合优化
"""

import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class YOLOv5TrainingManager:
    """YOLOv5训练管理器"""

    def __init__(self, project_root='D:/pythonPro/bishe'):
        self.project_root = Path(project_root)
        self.yolov5_dir = self.project_root / 'yolov5'  # 需要clone官方YOLOv5

    def train_with_config(self,
                          exp_name,
                          config_path,
                          data_yaml,
                          epochs=100,
                          batch_size=16,
                          img_size=640,
                          device='0',
                          loss_config=None):
        """
        使用指定配置训练YOLOv5

        参数:
            exp_name: 实验名称
            config_path: 模型配置文件路径
            data_yaml: 数据集配置文件
            epochs: 训练轮数
            batch_size: 批量大小
            img_size: 输入图像大小
            device: 设备ID
            loss_config: 损失函数配置 {'use_eiou': bool, 'use_focal': bool}
        """

        logger.info(f"\n{'=' * 70}")
        logger.info(f"🚀 开始训练: {exp_name}")
        logger.info(f"{'=' * 70}")

        if loss_config:
            logger.info(f"损失函数配置: {loss_config}")

        # YOLOv5官方训练命令
        cmd = [
            'python', 'train.py',
            '--img', str(img_size),
            '--batch', str(batch_size),
            '--epochs', str(epochs),
            '--data', str(data_yaml),
            '--cfg', str(config_path),
            '--weights', 'yolov5s.pt',
            '--device', device,
            '--name', exp_name,
            '--project', str(self.project_root / 'outputs' / 'ablation_study'),
            '--patience', '20',
            '--save-period', '10',
            '--cos-lr',
        ]

        # 添加损失函数参数
        if loss_config:
            if loss_config.get('use_eiou'):
                cmd.append('--use-eiou')
            if loss_config.get('use_focal'):
                cmd.append('--use-focal')

        try:
            # 在YOLOv5目录中运行
            result = subprocess.run(cmd, check=True, cwd=str(self.yolov5_dir))
            logger.info(f"✅ {exp_name} 训练完成")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {exp_name} 训练失败: {e}")
            return False

    def run_ablation_experiments(self):
        """运行完整的消融实验"""

        data_yaml = self.project_root / 'data' / 'processed' / 'flir' / 'dataset.yaml'

        experiments = [
            {
                'name': 'exp1_baseline',
                'description': 'Baseline: 原生YOLOv5s (标准损失)',
                'config': self.project_root / 'models' / 'yolov5' / 'configs' / 'yolov5s_thermal_exp1_baseline.yaml',
                'loss_config': {
                    'use_eiou': False,
                    'use_focal': False,
                }
            },
            {
                'name': 'exp2_lightweight',
                'description': 'Exp2: + 轻量化骨干网络 (GhostC3)',
                'config': self.project_root / 'models' / 'yolov5' / 'configs' / 'yolov5s_thermal_exp2_lightweight.yaml',
                'loss_config': {
                    'use_eiou': False,
                    'use_focal': False,
                }
            },
            {
                'name': 'exp3_eiou_loss',
                'description': 'Exp3: + EIoU Loss优化',
                'config': self.project_root / 'models' / 'yolov5' / 'configs' / 'yolov5s_thermal_exp2_lightweight.yaml',
                'loss_config': {
                    'use_eiou': True,
                    'use_focal': False,
                }
            },
            {
                'name': 'exp4_focal_loss',
                'description': 'Exp4: + Focal Loss优化',
                'config': self.project_root / 'models' / 'yolov5' / 'configs' / 'yolov5s_thermal_exp2_lightweight.yaml',
                'loss_config': {
                    'use_eiou': True,
                    'use_focal': True,
                }
            },
            {
                'name': 'exp5_attention',
                'description': 'Exp5: + 注意力机制 (CoordAttention)',
                'config': self.project_root / 'models' / 'yolov5' / 'configs' / 'yolov5s_thermal_exp3_attention.yaml',
                'loss_config': {
                    'use_eiou': True,
                    'use_focal': True,
                }
            },
        ]

        logger.info(f"🔬 开始热红外目标检测消融实验")
        logger.info(f"共有 {len(experiments)} 个实验")

        results = {}

        for i, exp in enumerate(experiments, 1):
            logger.info(f"\n[{i}/{len(experiments)}] {exp['description']}")

            success = self.train_with_config(
                exp_name=exp['name'],
                config_path=exp['config'],
                data_yaml=data_yaml,
                epochs=100,
                batch_size=16,
                loss_config=exp['loss_config'],
            )

            results[exp['name']] = {
                'description': exp['description'],
                'status': 'completed' if success else 'failed',
                'loss_config': exp['loss_config'],
            }

        # 保存结果
        result_file = self.project_root / 'outputs' / 'ablation_study' / 'experiment_log.json'
        result_file.parent.mkdir(parents=True, exist_ok=True)

        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✅ 所有消融实验完成！")
        logger.info(f"📂 结果已保存至: {self.project_root / 'outputs' / 'ablation_study'}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv5热红外训练 - 支持优化损失')
    parser.add_argument('--project-root', type=str, default='D:/pythonPro/bishe')
    parser.add_argument('--ablation', action='store_true', help='运行完整消融实验')

    args = parser.parse_args()

    manager = YOLOv5TrainingManager(args.project_root)

    if args.ablation:
        manager.run_ablation_experiments()
    else:
        logger.info("使用 --ablation 参数运行完整消融实验")


if __name__ == '__main__':
    main()