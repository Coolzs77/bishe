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
        self.detector路径 = args.detector
        self.video_path = args.video
        self.output路径 = Path(args.output)
        self.output路径.parent.mkdir(parents=True, exist_ok=True)
        
        # 解析tracker列表
        self.tracker列表 = [tracker.strip() for tracker in args.trackers.split(',')]
    
    def run_single_tracker_eval(self, trackername):
        """
        run单个tracker的evaluate
        
        参数:
            trackername: trackername
        
        返回:
            evaluateresults字典
        """
        print(f'\nevaluatetracker: {trackername}')
        
        # 构建evaluatecommand
        command = [
            'python', 'scripts/evaluate/eval_tracking.py',
            '--detector', self.detector路径,
            '--tracker', trackername,
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
        results文件 = Path(f'outputs/results/tracking_{trackername}/tracking_results.json')
        
        if results文件.exists():
            with open(results文件, 'r', encoding='utf-8') as f:
                evaluateresults = json.load(f)
            metrics = evaluateresults.get('overall_metrics', {})
        else:
            metrics = {}
        
        return {
            'tracker': trackername,
            'success': success,
            'duration': duration,
            'metrics': metrics,
        }
    
    def generate_comparison_report(self, all_results):
        """
        生成对比报告
        
        参数:
            所有results: 所有tracker的evaluateresults
        """
        # 生成CSV格式报告
        csv内容 = ['tracker,MOTA,IDF1,IDSW,MOTP,FP,FN,duration(秒)\n']
        
        for results in all_results:
            metrics = results.get('metrics', {})
            行 = [
                results['tracker'],
                str(metrics.get('MOTA', 'N/A')),
                str(metrics.get('IDF1', 'N/A')),
                str(metrics.get('IDSW', 'N/A')),
                str(metrics.get('MOTP', 'N/A')),
                str(metrics.get('FP', 'N/A')),
                str(metrics.get('FN', 'N/A')),
                f"{results.get('duration', 0):.1f}",
            ]
            csv内容.append(','.join(行) + '\n')
        
        # 保存CSV
        with open(self.output路径, 'w', encoding='utf-8') as f:
            f.writelines(csv内容)
        
        print(f'\nCSV报告已保存到: {self.output路径}')
        
        # 生成Markdown报告
        md路径 = self.output路径.with_suffix('.md')
        md内容 = ['# 跟踪算法对比报告\n\n']
        md内容.append(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        md内容.append(f'detector: {self.detector路径}\n\n')
        md内容.append('## 性能对比\n\n')
        md内容.append('| tracker | MOTA | IDF1 | IDSW | MOTP | FP | FN | duration(秒) |\n')
        md内容.append('|--------|------|------|------|------|----|----|----------|\n')
        
        for results in all_results:
            metrics = results.get('metrics', {})
            行 = f"| {results['tracker']} | "
            行 += f"{metrics.get('MOTA', 'N/A')} | "
            行 += f"{metrics.get('IDF1', 'N/A')} | "
            行 += f"{metrics.get('IDSW', 'N/A')} | "
            行 += f"{metrics.get('MOTP', 'N/A')} | "
            行 += f"{metrics.get('FP', 'N/A')} | "
            行 += f"{metrics.get('FN', 'N/A')} | "
            行 += f"{results.get('duration', 0):.1f} |\n"
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
        json路径 = self.output路径.with_suffix('.json')
        with open(json路径, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'detector': self.detector路径,
                'video': self.video_path,
                'results': all_results,
            }, f, indent=2, ensure_ascii=False)
        
        print(f'JSON汇总已保存到: {json路径}')
    
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
            mota排名 = sorted(all_results, key=lambda x: x.get('overall_metrics', {}).get('MOTA', 0) or 0, reverse=True)
            if mota排名:
                print(f'  - 综合性能最优 (MOTA): {mota排名[0]["tracker"]} ({mota排名[0].get("overall_metrics", {}).get("MOTA", "N/A")})')
            
            # 按IDF1排序找出身份保持最优
            idf1排名 = sorted(all_results, key=lambda x: x.get('overall_metrics', {}).get('IDF1', 0) or 0, reverse=True)
            if idf1排名:
                print(f'  - 身份保持最优 (IDF1): {idf1排名[0]["tracker"]} ({idf1排名[0].get("overall_metrics", {}).get("IDF1", "N/A")})')
            
            # 按速度排序找出最快
            速度排名 = sorted(all_results, key=lambda x: x.get('duration', float('inf')))
            if 速度排名:
                print(f'  - 速度最快: {速度排名[0]["tracker"]} ({速度排名[0].get("duration", "N/A"):.2f}s)')
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
        print(f'detector: {self.detector路径}')
        print(f'video_path: {self.video_path}')
        print(f'tracker: {", ".join(self.tracker列表)}')
        
        # run每个tracker的evaluate
        all_results = []
        for tracker in self.tracker列表:
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
    
    对比器 = TrackerComparator(args)
    对比器.run()


if __name__ == '__main__':
    main()
