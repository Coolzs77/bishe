#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跟踪算法对比脚本
对比不同跟踪算法在红外场景下的性能
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import subprocess


def 解析参数():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='对比不同跟踪算法的性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python compare_trackers.py --detector outputs/weights/best.pt --video data/processed/kaist/test_sequences/
  python compare_trackers.py --detector outputs/weights/best.pt --video data/processed/kaist/test_sequences/ --trackers deepsort,bytetrack
        '''
    )
    
    parser.add_argument('--detector', type=str, required=True,
                        help='检测器权重路径')
    parser.add_argument('--video', type=str, required=True,
                        help='测试视频路径')
    parser.add_argument('--trackers', type=str, default='deepsort,bytetrack,centertrack',
                        help='要对比的跟踪器列表，逗号分隔')
    parser.add_argument('--output', type=str, default='outputs/results/tracking_comparison.csv',
                        help='对比结果输出路径')
    parser.add_argument('--scenarios', type=str, default='all',
                        help='测试场景')
    
    return parser.parse_args()


class 跟踪器对比器:
    """跟踪算法对比器类"""
    
    def __init__(self, args):
        """
        初始化对比器
        
        参数:
            args: 命令行参数
        """
        self.args = args
        self.检测器路径 = args.detector
        self.视频路径 = args.video
        self.输出路径 = Path(args.output)
        self.输出路径.parent.mkdir(parents=True, exist_ok=True)
        
        # 解析跟踪器列表
        self.跟踪器列表 = [跟踪器.strip() for 跟踪器 in args.trackers.split(',')]
    
    def 运行单个跟踪器评估(self, 跟踪器名称):
        """
        运行单个跟踪器的评估
        
        参数:
            跟踪器名称: 跟踪器名称
        
        返回:
            评估结果字典
        """
        print(f'\n评估跟踪器: {跟踪器名称}')
        
        # 构建评估命令
        命令 = [
            'python', 'scripts/evaluate/eval_tracking.py',
            '--detector', self.检测器路径,
            '--tracker', 跟踪器名称,
            '--video', self.视频路径,
        ]
        
        开始时间 = datetime.now()
        
        try:
            结果 = subprocess.run(命令, capture_output=True, text=True, timeout=3600)
            成功 = 结果.returncode == 0
        except subprocess.TimeoutExpired:
            成功 = False
        except Exception as e:
            print(f'错误: {e}')
            成功 = False
        
        结束时间 = datetime.now()
        耗时 = (结束时间 - 开始时间).total_seconds()
        
        # 加载评估结果
        结果文件 = Path(f'outputs/results/tracking_{跟踪器名称}/tracking_results.json')
        
        if 结果文件.exists():
            with open(结果文件, 'r', encoding='utf-8') as f:
                评估结果 = json.load(f)
            指标 = 评估结果.get('overall_metrics', {})
        else:
            指标 = {}
        
        return {
            'tracker': 跟踪器名称,
            'success': 成功,
            'duration': 耗时,
            'metrics': 指标,
        }
    
    def 生成对比报告(self, 所有结果):
        """
        生成对比报告
        
        参数:
            所有结果: 所有跟踪器的评估结果
        """
        # 生成CSV格式报告
        csv内容 = ['跟踪器,MOTA,IDF1,IDSW,MOTP,FP,FN,耗时(秒)\n']
        
        for 结果 in 所有结果:
            指标 = 结果.get('metrics', {})
            行 = [
                结果['tracker'],
                str(指标.get('MOTA', 'N/A')),
                str(指标.get('IDF1', 'N/A')),
                str(指标.get('IDSW', 'N/A')),
                str(指标.get('MOTP', 'N/A')),
                str(指标.get('FP', 'N/A')),
                str(指标.get('FN', 'N/A')),
                f"{结果.get('duration', 0):.1f}",
            ]
            csv内容.append(','.join(行) + '\n')
        
        # 保存CSV
        with open(self.输出路径, 'w', encoding='utf-8') as f:
            f.writelines(csv内容)
        
        print(f'\nCSV报告已保存到: {self.输出路径}')
        
        # 生成Markdown报告
        md路径 = self.输出路径.with_suffix('.md')
        md内容 = ['# 跟踪算法对比报告\n\n']
        md内容.append(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        md内容.append(f'检测器: {self.检测器路径}\n\n')
        md内容.append('## 性能对比\n\n')
        md内容.append('| 跟踪器 | MOTA | IDF1 | IDSW | MOTP | FP | FN | 耗时(秒) |\n')
        md内容.append('|--------|------|------|------|------|----|----|----------|\n')
        
        for 结果 in 所有结果:
            指标 = 结果.get('metrics', {})
            行 = f"| {结果['tracker']} | "
            行 += f"{指标.get('MOTA', 'N/A')} | "
            行 += f"{指标.get('IDF1', 'N/A')} | "
            行 += f"{指标.get('IDSW', 'N/A')} | "
            行 += f"{指标.get('MOTP', 'N/A')} | "
            行 += f"{指标.get('FP', 'N/A')} | "
            行 += f"{指标.get('FN', 'N/A')} | "
            行 += f"{结果.get('duration', 0):.1f} |\n"
            md内容.append(行)
        
        md内容.append('\n## 分析\n\n')
        md内容.append('### 各算法特点\n\n')
        md内容.append('- **DeepSORT**: 基于卡尔曼滤波和ReID特征的跟踪，对外观特征依赖较大\n')
        md内容.append('- **ByteTrack**: 双阶段关联策略，不依赖外观特征，计算效率高\n')
        md内容.append('- **CenterTrack**: 基于中心点偏移的端到端跟踪，对快速运动敏感\n')
        
        with open(md路径, 'w', encoding='utf-8') as f:
            f.writelines(md内容)
        
        print(f'Markdown报告已保存到: {md路径}')
        
        # 保存JSON汇总
        json路径 = self.输出路径.with_suffix('.json')
        with open(json路径, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'detector': self.检测器路径,
                'video': self.视频路径,
                'results': 所有结果,
            }, f, indent=2, ensure_ascii=False)
        
        print(f'JSON汇总已保存到: {json路径}')
    
    def 打印对比结果(self, 所有结果):
        """打印对比结果"""
        print('\n' + '=' * 80)
        print('跟踪算法对比结果')
        print('=' * 80)
        
        # 表头
        print(f"{'跟踪器':<12} {'MOTA':>8} {'IDF1':>8} {'IDSW':>8} {'耗时(秒)':>10}")
        print('-' * 50)
        
        for 结果 in 所有结果:
            指标 = 结果.get('metrics', {})
            mota = 指标.get('MOTA', 'N/A')
            idf1 = 指标.get('IDF1', 'N/A')
            idsw = 指标.get('IDSW', 'N/A')
            耗时 = 结果.get('duration', 0)
            
            print(f"{结果['tracker']:<12} {str(mota):>8} {str(idf1):>8} {str(idsw):>8} {耗时:>10.1f}")
        
        # 找出最优
        print('\n最优跟踪器分析:')
        # TODO: 根据实际指标找出最优
        print('  - 综合性能最优: 待评估')
        print('  - 身份保持最优: 待评估')
        print('  - 速度最快: 待评估')
    
    def 运行(self):
        """运行对比流程"""
        print('=' * 60)
        print('跟踪算法对比')
        print('=' * 60)
        print(f'检测器: {self.检测器路径}')
        print(f'视频路径: {self.视频路径}')
        print(f'跟踪器: {", ".join(self.跟踪器列表)}')
        
        # 运行每个跟踪器的评估
        所有结果 = []
        for 跟踪器 in self.跟踪器列表:
            结果 = self.运行单个跟踪器评估(跟踪器)
            所有结果.append(结果)
        
        # 打印对比结果
        self.打印对比结果(所有结果)
        
        # 生成报告
        self.生成对比报告(所有结果)
        
        print('\n' + '=' * 60)
        print('对比完成!')
        print('=' * 60)


def main():
    """主函数"""
    args = 解析参数()
    
    对比器 = 跟踪器对比器(args)
    对比器.运行()


if __name__ == '__main__':
    main()
