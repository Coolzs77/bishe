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
    """解析command行参数"""
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
                        help='detectorweights_path')
    parser.add_argument('--video', type=str, required=True,
                        help='测试video_path')
    parser.add_argument('--trackers', type=str, default='deepsort,bytetrack,centertrack',
                        help='要对比的tracker列表，逗号分隔')
    parser.add_argument('--output', type=str, default='outputs/results/tracking_comparison.csv',
                        help='对比resultsoutput路径')
    parser.add_argument('--scenarios', type=str, default='all',
                        help='测试场景')
    
    return parser.parse_args()


class TrackerComparator:
    """跟踪算法对比器类"""
    
    def __init__(self, args):
        """
        初始化对比器
        
        参数:
            args: command行参数
        """
        self.args = args
        self.detector_path = args.detector
        self.video_path = args.video
        self.output_path = Path(args.output)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 解析tracker列表
        self.tracker_list = [tracker.strip() for tracker in args.trackers.split(',')]
    
    def run_single_tracker_eval(self, tracker_name):
        """
        run单个tracker的evaluate
        
        参数:
            tracker_name: trackername
        
        返回:
            evaluateresults字典
        """
        print(f'\nevaluatetracker: {tracker_name}')
        
        # 构建evaluatecommand
        command = [
            'python', 'scripts/evaluate/eval_tracking.py',
            '--detector', self.detector_path,
            '--tracker', tracker_name,
            '--video', self.video_path,
        ]
        
        start_time = datetime.now()
        
        try:
            results = subprocess.run(command, capture_output=True, text=True, timeout=3600)
            success = results.returncode == 0
        except subprocess.TimeoutExpired:
            success = False
        except Exception as e:
            print(f'错误: {e}')
            success = False
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 加载evaluateresults
        results_file = Path(f'outputs/results/tracking_{tracker_name}/tracking_results.json')
        
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                eval_results = json.load(f)
            metrics = eval_results.get('overall_metrics', {})
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
            all_results: 所有tracker的evaluateresults
        """
        # 生成CSV格式报告
        csv_content = ['tracker,MOTA,IDF1,IDSW,MOTP,FP,FN,duration(秒)\n']
        
        for results in all_results:
            metrics = results.get('metrics', {})
            row = [
                results['tracker'],
                str(metrics.get('MOTA', 'N/A')),
                str(metrics.get('IDF1', 'N/A')),
                str(metrics.get('IDSW', 'N/A')),
                str(metrics.get('MOTP', 'N/A')),
                str(metrics.get('FP', 'N/A')),
                str(metrics.get('FN', 'N/A')),
                f"{results.get('duration', 0):.1f}",
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
        md_content.append(f'detector: {self.detector_path}\n\n')
        md_content.append('## 性能对比\n\n')
        md_content.append('| tracker | MOTA | IDF1 | IDSW | MOTP | FP | FN | duration(秒) |\n')
        md_content.append('|--------|------|------|------|------|----|----|----------|\n')
        
        for results in all_results:
            metrics = results.get('metrics', {})
            row = f"| {results['tracker']} | "
            row += f"{metrics.get('MOTA', 'N/A')} | "
            row += f"{metrics.get('IDF1', 'N/A')} | "
            row += f"{metrics.get('IDSW', 'N/A')} | "
            row += f"{metrics.get('MOTP', 'N/A')} | "
            row += f"{metrics.get('FP', 'N/A')} | "
            row += f"{metrics.get('FN', 'N/A')} | "
            row += f"{results.get('duration', 0):.1f} |\n"
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
        """print_comparison_results"""
        print('\n' + '=' * 80)
        print('跟踪算法对比results')
        print('=' * 80)
        
        # 表头
        print(f"{'tracker':<12} {'MOTA':>8} {'IDF1':>8} {'IDSW':>8} {'duration(秒)':>10}")
        print('-' * 50)
        
        for results in all_results:
            metrics = results.get('metrics', {})
            mota = metrics.get('MOTA', 'N/A')
            idf1 = metrics.get('IDF1', 'N/A')
            idsw = metrics.get('IDSW', 'N/A')
            duration = results.get('duration', 0)
            
            print(f"{results['tracker']:<12} {str(mota):>8} {str(idf1):>8} {str(idsw):>8} {duration:>10.1f}")
        
        # 找出最优
        print('\n最优tracker分析:')
        
        try:
            # 按MOTA排序找出最优
            mota_ranking = sorted(all_results, key=lambda x: x.get('overall_metrics', {}).get('MOTA', 0) or 0, reverse=True)
            if mota_ranking:
                print(f'  - 综合性能最优 (MOTA): {mota_ranking[0]["tracker"]} ({mota_ranking[0].get("overall_metrics", {}).get("MOTA", "N/A")})')
            
            # 按IDF1排序找出身份保持最优
            idf1_ranking = sorted(all_results, key=lambda x: x.get('overall_metrics', {}).get('IDF1', 0) or 0, reverse=True)
            if idf1_ranking:
                print(f'  - 身份保持最优 (IDF1): {idf1_ranking[0]["tracker"]} ({idf1_ranking[0].get("overall_metrics", {}).get("IDF1", "N/A")})')
            
            # 按速度排序找出最快
            speed_ranking = sorted(all_results, key=lambda x: x.get('duration', float('inf')))
            if speed_ranking:
                print(f'  - 速度最快: {speed_ranking[0]["tracker"]} ({speed_ranking[0].get("duration", "N/A"):.2f}s)')
        except Exception as e:
            print(f'  分析出错: {e}')
            print('  - 综合性能最优: 待evaluate')
            print('  - 身份保持最优: 待evaluate')
            print('  - 速度最快: 待evaluate')
    
    def run(self):
        """run对比流程"""
        print('=' * 60)
        print('跟踪算法对比')
        print('=' * 60)
        print(f'detector: {self.detector_path}')
        print(f'video_path: {self.video_path}')
        print(f'tracker: {", ".join(self.tracker_list)}')
        
        # run每个tracker的evaluate
        all_results = []
        for tracker in self.tracker_list:
            results = self.run_single_tracker_eval(tracker)
            all_results.append(results)
        
        # print_comparison_results
        self.print_comparison_results(all_results)
        
        # generate_report
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
