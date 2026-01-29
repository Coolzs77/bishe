#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验脚本
自动运行一系列消融实验，验证各模块的效果
"""

import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
import json


def 解析参数():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='运行消融实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python ablation_study.py
  python ablation_study.py --experiments backbone,attention
  python ablation_study.py --epochs 50 --skip-existing
        '''
    )
    
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='基础配置文件')
    parser.add_argument('--output', type=str, default='outputs/results/ablation',
                        help='结果输出目录')
    parser.add_argument('--experiments', type=str, default='all',
                        help='要运行的实验，逗号分隔 (all/baseline/backbone/loss/attention/combined)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='每个实验的训练轮数')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已完成的实验')
    
    return parser.parse_args()


# 消融实验配置
实验配置列表 = [
    {
        'name': 'baseline',
        'description': '基准模型 (YOLOv5s原生配置)',
        'backbone': 'c3',
        'loss': 'ciou',
        'attention': 'none',
    },
    {
        'name': 'backbone_ghost',
        'description': 'Ghost-C3骨干网络',
        'backbone': 'ghost',
        'loss': 'ciou',
        'attention': 'none',
    },
    {
        'name': 'backbone_shuffle',
        'description': 'Shuffle-C3骨干网络',
        'backbone': 'shuffle',
        'loss': 'ciou',
        'attention': 'none',
    },
    {
        'name': 'loss_siou',
        'description': 'SIoU损失函数',
        'backbone': 'c3',
        'loss': 'siou',
        'attention': 'none',
    },
    {
        'name': 'loss_eiou',
        'description': 'EIoU损失函数',
        'backbone': 'c3',
        'loss': 'eiou',
        'attention': 'none',
    },
    {
        'name': 'attention_cbam',
        'description': 'CBAM注意力机制',
        'backbone': 'c3',
        'loss': 'ciou',
        'attention': 'cbam',
    },
    {
        'name': 'attention_coordatt',
        'description': '坐标注意力机制',
        'backbone': 'c3',
        'loss': 'ciou',
        'attention': 'coordatt',
    },
    {
        'name': 'attention_se',
        'description': 'SE注意力机制',
        'backbone': 'c3',
        'loss': 'ciou',
        'attention': 'se',
    },
]


class 消融实验管理器:
    """消融实验管理器类"""
    
    def __init__(self, args):
        """
        初始化管理器
        
        参数:
            args: 命令行参数
        """
        self.args = args
        self.输出目录 = Path(args.output)
        self.输出目录.mkdir(parents=True, exist_ok=True)
        
        # 结果记录
        self.实验结果 = []
    
    def 获取实验列表(self):
        """根据参数获取要运行的实验列表"""
        if self.args.experiments == 'all':
            return 实验配置列表
        
        实验名称列表 = [名称.strip() for 名称 in self.args.experiments.split(',')]
        
        筛选后列表 = []
        for 实验 in 实验配置列表:
            # 检查实验名称或类别是否匹配
            if 实验['name'] in 实验名称列表:
                筛选后列表.append(实验)
            elif any(实验['name'].startswith(名称) for 名称 in 实验名称列表):
                筛选后列表.append(实验)
        
        return 筛选后列表
    
    def 检查实验是否完成(self, 实验名称):
        """检查实验是否已完成"""
        结果文件 = self.输出目录 / f'{实验名称}_results.json'
        return 结果文件.exists()
    
    def 运行单个实验(self, 实验配置):
        """
        运行单个消融实验
        
        参数:
            实验配置: 实验配置字典
        
        返回:
            实验结果字典
        """
        实验名称 = 实验配置['name']
        
        print(f'\n{"="*60}')
        print(f'实验: {实验名称}')
        print(f'描述: {实验配置["description"]}')
        print(f'{"="*60}')
        
        # 检查是否跳过
        if self.args.skip_existing and self.检查实验是否完成(实验名称):
            print(f'实验 {实验名称} 已完成，跳过')
            return None
        
        # 构建训练命令
        命令 = [
            'python', 'scripts/train/train_yolov5.py',
            '--name', 实验名称,
            '--epochs', str(self.args.epochs),
            '--backbone', 实验配置['backbone'],
            '--loss', 实验配置['loss'],
            '--attention', 实验配置['attention'],
        ]
        
        print(f'执行命令: {" ".join(命令)}')
        
        # 运行训练
        开始时间 = datetime.now()
        
        try:
            结果 = subprocess.run(命令, capture_output=True, text=True)
            成功 = 结果.returncode == 0
            输出 = 结果.stdout
            错误 = 结果.stderr
        except Exception as e:
            成功 = False
            输出 = ''
            错误 = str(e)
        
        结束时间 = datetime.now()
        耗时 = (结束时间 - 开始时间).total_seconds()
        
        # 记录结果
        实验结果 = {
            'name': 实验名称,
            'description': 实验配置['description'],
            'config': 实验配置,
            'success': 成功,
            'duration_seconds': 耗时,
            'start_time': 开始时间.isoformat(),
            'end_time': 结束时间.isoformat(),
            'output': 输出[-5000:] if 输出 else '',  # 只保留最后5000字符
            'error': 错误[-2000:] if 错误 else '',
            'metrics': self._extract_metrics_from_log(输出)
        }
        
        # 保存单个实验结果
        结果文件 = self.输出目录 / f'{实验名称}_results.json'
        with open(结果文件, 'w', encoding='utf-8') as f:
            json.dump(实验结果, f, indent=2, ensure_ascii=False)
        
        if 成功:
            print(f'实验 {实验名称} 完成，耗时 {耗时:.1f} 秒')
        else:
            print(f'实验 {实验名称} 失败')
            if 错误:
                print(f'错误信息: {错误[:500]}')
        
        return 实验结果
    
    def _extract_metrics_from_log(self, 日志内容):
        """
        从训练日志中提取指标
        
        参数:
            日志内容: 训练输出日志
        
        返回:
            指标字典
        """
        import re
        
        指标 = {
            'mAP_0.5': None,
            'mAP_0.5_0.95': None,
            'precision': None,
            'recall': None,
        }
        
        if not 日志内容:
            return 指标
        
        try:
            # 使用正则表达式提取常见指标
            # YOLOv5日志格式示例: "mAP@0.5: 0.856"
            
            mAP_05_match = re.search(r'mAP@?0\.5:?\s+([\d.]+)', 日志内容)
            if mAP_05_match:
                指标['mAP_0.5'] = float(mAP_05_match.group(1))
            
            mAP_05_095_match = re.search(r'mAP@?0\.5:0\.95:?\s+([\d.]+)', 日志内容)
            if mAP_05_095_match:
                指标['mAP_0.5_0.95'] = float(mAP_05_095_match.group(1))
            
            precision_match = re.search(r'[Pp]recision:?\s+([\d.]+)', 日志内容)
            if precision_match:
                指标['precision'] = float(precision_match.group(1))
            
            recall_match = re.search(r'[Rr]ecall:?\s+([\d.]+)', 日志内容)
            if recall_match:
                指标['recall'] = float(recall_match.group(1))
            
        except Exception as e:
            print(f'  提取指标失败: {e}')
        
        return 指标
    
    def 生成对比报告(self):
        """生成消融实验对比报告"""
        print('\n生成消融实验报告...')
        
        # 加载所有实验结果
        所有结果 = []
        for 实验 in 实验配置列表:
            结果文件 = self.输出目录 / f'{实验["name"]}_results.json'
            if 结果文件.exists():
                with open(结果文件, 'r', encoding='utf-8') as f:
                    所有结果.append(json.load(f))
        
        if not 所有结果:
            print('没有找到实验结果')
            return
        
        # 生成Markdown报告
        报告内容 = ['# 消融实验报告\n']
        报告内容.append(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        报告内容.append(f'实验数量: {len(所有结果)}\n\n')
        
        # 实验配置表格
        报告内容.append('## 实验配置\n\n')
        报告内容.append('| 实验名称 | 骨干网络 | 损失函数 | 注意力机制 | 状态 |\n')
        报告内容.append('|---------|---------|---------|-----------|------|\n')
        
        for 结果 in 所有结果:
            配置 = 结果.get('config', {})
            状态 = '✓' if 结果.get('success') else '✗'
            报告内容.append(f"| {结果['name']} | {配置.get('backbone', '-')} | "
                          f"{配置.get('loss', '-')} | {配置.get('attention', '-')} | {状态} |\n")
        
        # 性能指标表格
        报告内容.append('\n## 性能指标\n\n')
        报告内容.append('| 实验名称 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |\n')
        报告内容.append('|---------|---------|-------------|-----------|--------|\n')
        
        for 结果 in 所有结果:
            指标 = 结果.get('metrics', {})
            报告内容.append(f"| {结果['name']} | "
                          f"{指标.get('mAP_0.5', '-')} | "
                          f"{指标.get('mAP_0.5_0.95', '-')} | "
                          f"{指标.get('precision', '-')} | "
                          f"{指标.get('recall', '-')} |\n")
        
        # 耗时统计
        报告内容.append('\n## 训练耗时\n\n')
        报告内容.append('| 实验名称 | 耗时(秒) | 耗时(分钟) |\n')
        报告内容.append('|---------|---------|----------|\n')
        
        for 结果 in 所有结果:
            耗时秒 = 结果.get('duration_seconds', 0)
            耗时分 = 耗时秒 / 60 if 耗时秒 else 0
            报告内容.append(f"| {结果['name']} | {耗时秒:.1f} | {耗时分:.1f} |\n")
        
        # 保存报告
        报告路径 = self.输出目录 / 'ablation_report.md'
        with open(报告路径, 'w', encoding='utf-8') as f:
            f.writelines(报告内容)
        
        print(f'报告已保存到: {报告路径}')
        
        # 保存汇总JSON
        汇总路径 = self.输出目录 / 'ablation_summary.json'
        with open(汇总路径, 'w', encoding='utf-8') as f:
            json.dump(所有结果, f, indent=2, ensure_ascii=False)
        
        print(f'汇总结果已保存到: {汇总路径}')
    
    def 运行(self):
        """运行全部消融实验"""
        print('=' * 60)
        print('消融实验')
        print('=' * 60)
        print(f'输出目录: {self.输出目录}')
        print(f'每个实验训练轮数: {self.args.epochs}')
        
        # 获取实验列表
        实验列表 = self.获取实验列表()
        print(f'待运行实验数: {len(实验列表)}')
        
        for i, 实验 in enumerate(实验列表):
            print(f'\n[{i+1}/{len(实验列表)}] {实验["name"]}')
        
        print('\n开始运行实验...')
        
        # 运行每个实验
        for 实验 in 实验列表:
            结果 = self.运行单个实验(实验)
            if 结果:
                self.实验结果.append(结果)
        
        # 生成报告
        self.生成对比报告()
        
        print('\n' + '=' * 60)
        print('消融实验完成!')
        print('=' * 60)


def main():
    """主函数"""
    args = 解析参数()
    
    管理器 = 消融实验管理器(args)
    管理器.运行()


if __name__ == '__main__':
    main()
