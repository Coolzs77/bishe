#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验脚本
自动run一系列消融实验，验证各模块的效果
"""

import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
import json


def parse_args():
    """解析command行参数"""
    parser = argparse.ArgumentParser(
        description='run消融实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python ablation_study.py
  python ablation_study.py --experiments backbone,attention
  python ablation_study.py --epochs 50 --skip-existing
        '''
    )
    
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='基础config文件')
    parser.add_argument('--output', type=str, default='outputs/results/ablation',
                        help='resultsoutput目录')
    parser.add_argument('--experiments', type=str, default='all',
                        help='要run的实验，逗号分隔 (all/baseline/backbone/loss/attention/combined)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='每个实验的训练轮数')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已完成的实验')
    
    return parser.parse_args()


# 消融实验config
experiment_configs = [
    {
        'name': 'baseline',
        'description': '基准model (YOLOv5s原生config)',
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
        'description': 'SIoUloss函数',
        'backbone': 'c3',
        'loss': 'siou',
        'attention': 'none',
    },
    {
        'name': 'loss_eiou',
        'description': 'EIoUloss函数',
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


class AblationStudyManager:
    """AblationStudyManager类"""
    
    def __init__(self, args):
        """
        初始化管理器
        
        参数:
            args: command行参数
        """
        self.args = args
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # results记录
        self.experiment_results = []
    
    def get_experiment_list(self):
        """根据参数获取要run的实验列表"""
        if self.args.experiments == 'all':
            return experiment_configs
        
        experiment_name_list = [name.strip() for name in self.args.experiments.split(',')]
        
        filtered_list = []
        for experiment in experiment_configs:
            # 检查experiment_name或classes是否匹配
            if experiment['name'] in experiment_name_list:
                filtered_list.append(experiment)
            elif any(experiment['name'].startswith(name) for name in experiment_name_list):
                filtered_list.append(experiment)
        
        return filtered_list
    
    def check_experiment_complete(self, experiment_name):
        """检查实验是否已完成"""
        results_file = self.output_dir / f'{experiment_name}_results.json'
        return results_file.exists()
    
    def run_single_experiment(self, experiment_config):
        """
        run单个消融实验
        
        参数:
            experiment_config: 实验config字典
        
        返回:
            实验results字典
        """
        experiment_name = experiment_config['name']
        
        print(f'\n{"="*60}')
        print(f'实验: {experiment_name}')
        print(f'描述: {experiment_config["description"]}')
        print(f'{"="*60}')
        
        # 检查是否跳过
        if self.args.skip_existing and self.check_experiment_complete(experiment_name):
            print(f'实验 {experiment_name} 已完成，跳过')
            return None
        
        # 构建训练command
        command = [
            'python', 'scripts/train/train_yolov5.py',
            '--name', experiment_name,
            '--epochs', str(self.args.epochs),
            '--backbone', experiment_config['backbone'],
            '--loss', experiment_config['loss'],
            '--attention', experiment_config['attention'],
        ]
        
        print(f'执行command: {" ".join(command)}')
        
        # run训练
        start_time = datetime.now()
        
        try:
            results = subprocess.run(command, capture_output=True, text=True)
            success = results.returncode == 0
            output = results.stdout
            error = results.stderr
        except Exception as e:
            success = False
            output = ''
            error = str(e)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 记录results
        experiment_results = {
            'name': experiment_name,
            'description': experiment_config['description'],
            'config': experiment_config,
            'success': success,
            'duration_seconds': duration,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'output': output[-5000:] if output else '',  # 只保留最后5000字符
            'error': error[-2000:] if error else '',
            'metrics': self._extract_metrics_from_log(output)
        }
        
        # 保存单个实验results
        results_file = self.output_dir / f'{experiment_name}_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False)
        
        if success:
            print(f'实验 {experiment_name} 完成，duration {duration:.1f} 秒')
        else:
            print(f'实验 {experiment_name} 失败')
            if error:
                print(f'错误信息: {error[:500]}')
        
        return experiment_results
    
    def _extract_metrics_from_log(self, log_content):
        """
        从训练日志中提取metrics
        
        参数:
            log_content: 训练output日志
        
        返回:
            metrics字典
        """
        import re
        
        metrics = {
            'mAP_0.5': None,
            'mAP_0.5_0.95': None,
            'precision': None,
            'recall': None,
        }
        
        if not log_content:
            return metrics
        
        try:
            # 使用正则表达式提取常见metrics
            # YOLOv5日志格式示例: "mAP@0.5: 0.856"
            
            mAP_05_match = re.search(r'mAP@?0\.5:?\s+([\d.]+)', log_content)
            if mAP_05_match:
                metrics['mAP_0.5'] = float(mAP_05_match.group(1))
            
            mAP_05_095_match = re.search(r'mAP@?0\.5:0\.95:?\s+([\d.]+)', log_content)
            if mAP_05_095_match:
                metrics['mAP_0.5_0.95'] = float(mAP_05_095_match.group(1))
            
            precision_match = re.search(r'[Pp]recision:?\s+([\d.]+)', log_content)
            if precision_match:
                metrics['precision'] = float(precision_match.group(1))
            
            recall_match = re.search(r'[Rr]ecall:?\s+([\d.]+)', log_content)
            if recall_match:
                metrics['recall'] = float(recall_match.group(1))
            
        except Exception as e:
            print(f'  提取metrics失败: {e}')
        
        return metrics
    
    def generate_comparison_report(self):
        """生成消融实验对比报告"""
        print('\n生成消融实验报告...')
        
        # 加载所有实验results
        all_results = []
        for experiment in experiment_configs:
            results_file = self.output_dir / f'{experiment["name"]}_results.json'
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    all_results.append(json.load(f))
        
        if not all_results:
            print('没有找到实验results')
            return
        
        # 生成Markdown报告
        report_content = ['# 消融实验报告\n']
        report_content.append(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        report_content.append(f'实验count: {len(all_results)}\n\n')
        
        # 实验config表格
        report_content.append('## 实验config\n\n')
        report_content.append('| experiment_name | 骨干网络 | loss函数 | 注意力机制 | 状态 |\n')
        report_content.append('|---------|---------|---------|-----------|------|\n')
        
        for results in all_results:
            config = results.get('config', {})
            status = '✓' if results.get('success') else '✗'
            report_content.append(f"| {results['name']} | {config.get('backbone', '-')} | "
                          f"{config.get('loss', '-')} | {config.get('attention', '-')} | {status} |\n")
        
        # 性能metrics表格
        report_content.append('\n## 性能metrics\n\n')
        report_content.append('| experiment_name | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |\n')
        report_content.append('|---------|---------|-------------|-----------|--------|\n')
        
        for results in all_results:
            metrics = results.get('metrics', {})
            report_content.append(f"| {results['name']} | "
                          f"{metrics.get('mAP_0.5', '-')} | "
                          f"{metrics.get('mAP_0.5_0.95', '-')} | "
                          f"{metrics.get('precision', '-')} | "
                          f"{metrics.get('recall', '-')} |\n")
        
        # duration统计
        report_content.append('\n## 训练duration\n\n')
        report_content.append('| experiment_name | duration(秒) | duration(分钟) |\n')
        report_content.append('|---------|---------|----------|\n')
        
        for results in all_results:
            duration_sec = results.get('duration_seconds', 0)
            duration_min = duration_sec / 60 if duration_sec else 0
            report_content.append(f"| {results['name']} | {duration_sec:.1f} | {duration_min:.1f} |\n")
        
        # 保存报告
        report_path = self.output_dir / 'ablation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_content)
        
        print(f'报告已保存到: {report_path}')
        
        # 保存汇总JSON
        summary_path = self.output_dir / 'ablation_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f'summarize_results已保存到: {summary_path}')
    
    def run(self):
        """run全部消融实验"""
        print('=' * 60)
        print('消融实验')
        print('=' * 60)
        print(f'output目录: {self.output_dir}')
        print(f'每个实验训练轮数: {self.args.epochs}')
        
        # 获取实验列表
        experiment_list = self.get_experiment_list()
        print(f'待run_experiments数: {len(experiment_list)}')
        
        for i, experiment in enumerate(experiment_list):
            print(f'\n[{i+1}/{len(experiment_list)}] {experiment["name"]}')
        
        print('\n开始run_experiments...')
        
        # run每个实验
        for experiment in experiment_list:
            results = self.run_single_experiment(experiment)
            if results:
                self.experiment_results.append(results)
        
        # generate_report
        self.generate_comparison_report()
        
        print('\n' + '=' * 60)
        print('消融实验完成!')
        print('=' * 60)


def main():
    """主函数"""
    args = parse_args()
    
    manager = AblationStudyManager(args)
    manager.run()


if __name__ == '__main__':
    main()
