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


def parse_args():
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


class TrackerComparator:
    """跟踪算法对比器类"""
    
    def __init__(self, args):
        """
        初始化对比器
        
        参数:
            args: 命令行参数
        """
        self.args = args
        self.detector_path = args.detector
        self.video_path = args.video
        self.output_path = Path(args.output)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 解析跟踪器列表
        self.tracker_list = [tracker.strip() for tracker in args.trackers.split(',')]
    
    def run_single_tracker_eval(self, tracker_name):
        """
        运行单个跟踪器的评估
        
        参数:
            跟踪器名称: 跟踪器名称
        
        返回:
            评估结果字典
        """
        print(f'\n评估跟踪器: {tracker_name}')
        
        # 构建评估命令
        command = [
            'python', 'scripts/evaluate/eval_tracking.py',
            '--detector', self.detector_path,
            '--tracker', tracker_name,
            '--video', self.video_path,
        ]
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            success = False
        except Exception as e:
            print(f'错误: {e}')
            success = False
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 加载评估结果
        result_file = Path(f'outputs/results/tracking_{tracker_name}/tracking_results.json')
        
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                eval_result = json.load(f)
            metrics = eval_result.get('overall_metrics', {})
        else:
            metrics = {}
        
        return {
            'tracker': tracker_name,
            'success': success,
            'duration': duration,
            'metrics': metrics,
        }
    
    def generate_comparison_report(self, all_results):
        """
        生成对比报告
        
        参数:
            所有结果: 所有跟踪器的评估结果
        """
        # 生成CSV格式报告
        csv_content = ['跟踪器,MOTA,IDF1,IDSW,MOTP,FP,FN,耗时(秒)\n']
        
        for result in all_results:
            metrics = result.get('metrics', {})
            row = [
                result['tracker'],
                str(metrics.get('MOTA', 'N/A')),
                str(metrics.get('IDF1', 'N/A')),
                str(metrics.get('IDSW', 'N/A')),
                str(metrics.get('MOTP', 'N/A')),
                str(metrics.get('FP', 'N/A')),
                str(metrics.get('FN', 'N/A')),
                f"{result.get('duration', 0):.1f}",
            ]
            csv_content.append(','.join(row) + '\n')
        
        # 保存CSV
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.writelines(csv_content)
        
        print(f'\nCSV报告已保存到: {self.output_path}')
        
        # 生成Markdown报告
        md_path = self.output_path.with_suffix('.md')
        md_content = ['# 跟踪算法对比报告\n\n']
        md_content.append(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        md_content.append(f'检测器: {self.detector_path}\n\n')
        md_content.append('## 性能对比\n\n')
        md_content.append('| 跟踪器 | MOTA | IDF1 | IDSW | MOTP | FP | FN | 耗时(秒) |\n')
        md_content.append('|--------|------|------|------|------|----|----|----------|\n')
        
        for result in all_results:
            metrics = result.get('metrics', {})
            row = f"| {result['tracker']} | "
            row += f"{metrics.get('MOTA', 'N/A')} | "
            row += f"{metrics.get('IDF1', 'N/A')} | "
            row += f"{metrics.get('IDSW', 'N/A')} | "
            row += f"{metrics.get('MOTP', 'N/A')} | "
            row += f"{metrics.get('FP', 'N/A')} | "
            row += f"{metrics.get('FN', 'N/A')} | "
            row += f"{result.get('duration', 0):.1f} |\n"
            md_content.append(row)
        
        md_content.append('\n## 分析\n\n')
        md_content.append('### 各算法特点\n\n')
        md_content.append('- **DeepSORT**: 基于卡尔曼滤波和ReID特征的跟踪，对外观特征依赖较大\n')
        md_content.append('- **ByteTrack**: 双阶段关联策略，不依赖外观特征，计算效率高\n')
        md_content.append('- **CenterTrack**: 基于中心点偏移的端到端跟踪，对快速运动敏感\n')
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.writelines(md_content)
        
        print(f'Markdown报告已保存到: {md_path}')
        
        # 保存JSON汇总
        json_path = self.output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'detector': self.detector_path,
                'video': self.video_path,
                'results': all_results,
            }, f, indent=2, ensure_ascii=False)
        
        print(f'JSON汇总已保存到: {json_path}')
    
    def print_comparison_results(self, all_results):
        """打印对比结果"""
        print('\n' + '=' * 80)
        print('跟踪算法对比结果')
        print('=' * 80)
        
        # 表头
        print(f"{'跟踪器':<12} {'MOTA':>8} {'IDF1':>8} {'IDSW':>8} {'耗时(秒)':>10}")
        print('-' * 50)
        
        for result in all_results:
            metrics = result.get('metrics', {})
            mota = metrics.get('MOTA', 'N/A')
            idf1 = metrics.get('IDF1', 'N/A')
            idsw = metrics.get('IDSW', 'N/A')
            duration = result.get('duration', 0)
            
            print(f"{result['tracker']:<12} {str(mota):>8} {str(idf1):>8} {str(idsw):>8} {duration:>10.1f}")
        
        # 找出最优
        print('\n最优跟踪器分析:')
        # TODO: 根据实际指标找出最优
        print('  - 综合性能最优: 待评估')
        print('  - 身份保持最优: 待评估')
        print('  - 速度最快: 待评估')
    
    def run(self):
        """运行对比流程"""
        print('=' * 60)
        print('跟踪算法对比')
        print('=' * 60)
        print(f'检测器: {self.detector_path}')
        print(f'视频路径: {self.video_path}')
        print(f'跟踪器: {", ".join(self.tracker_list)}')
        
        # 运行每个跟踪器的评估
        all_results = []
        for tracker in self.tracker_list:
            result = self.run_single_tracker_eval(tracker)
            all_results.append(result)
        
        # 打印对比结果
        self.print_comparison_results(all_results)
        
        # 生成报告
        self.generate_comparison_report(all_results)
        
        print('\n' + '=' * 60)
        print('对比完成!')
        print('=' * 60)


def main():
    """主函数"""
    args = parse_args()
    
    comparator = TrackerComparator(args)
    comparator.run()


if __name__ == '__main__':
    main()
