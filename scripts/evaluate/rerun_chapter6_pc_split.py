#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import yaml


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]


SEQ_PREFIXES = {
    'seq006': 'ZAtDSNuZZjkZFvMAo',
    'seq009': 't3f7QC8hZr6zYXpEZ',
}

MODELS = {
    'baseline': ROOT / 'outputs' / 'ablation_study' / 'ablation_exp01_baseline' / 'weights' / 'best.pt',
    'eiou': ROOT / 'outputs' / 'ablation_study' / 'ablation_exp07_eiou' / 'weights' / 'best.pt',
    'ghost_eiou': ROOT / 'outputs' / 'ablation_study' / 'ablation_exp09_ghost_eiou' / 'weights' / 'best.pt',
}


def ensure_sequence_subsets() -> dict[str, Path]:
    from scripts.data.create_flir_sequence_subsets import main as create_subsets_main
    import sys

    output_dir = ROOT / 'data' / 'processed' / 'flir' / 'sequence_eval'
    expected = {seq: output_dir / f'{seq}_dataset.yaml' for seq in SEQ_PREFIXES}
    if all(path.exists() for path in expected.values()):
        return expected

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            'create_flir_sequence_subsets.py',
            '--subset', f"seq006={SEQ_PREFIXES['seq006']}",
            '--subset', f"seq009={SEQ_PREFIXES['seq009']}",
        ]
        create_subsets_main()
    finally:
        sys.argv = original_argv

    return expected


def load_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def make_detection_args(weights: Path, dataset_yaml: Path, output_dir: Path) -> Namespace:
    config = load_yaml(ROOT / 'configs' / 'eval_detection.yaml')
    runtime = config.get('runtime', {})
    return Namespace(
        config=str(ROOT / 'configs' / 'eval_detection.yaml'),
        mode='metric',
        weights=str(weights),
        data=str(dataset_yaml),
        batch_size=runtime.get('batch_size', 16),
        img_size=runtime.get('img_size', 640),
        conf_thres=runtime.get('conf_thres', 0.001),
        iou_thres=runtime.get('iou_thres', 0.6),
        device=str(runtime.get('device', '0')),
        workers=0,
        task='test',
        batch_eval=False,
        ablation_dir=None,
        stage='all',
        weights_name='best.pt',
        sort_by='map5095',
        save_csv=False,
        save_json=True,
        output=str(output_dir),
    )


def run_detection(output_root: Path) -> dict:
    from scripts.evaluate.eval_detection import DetectionEvaluator

    subset_yamls = ensure_sequence_subsets()
    detection_rows = []
    for model_name, weights in MODELS.items():
        for seq_name, dataset_yaml in subset_yamls.items():
            run_dir = output_root / model_name / seq_name
            args = make_detection_args(weights, dataset_yaml, run_dir)
            evaluator = DetectionEvaluator(args)
            result = evaluator.evaluate_metric()
            evaluator.save_results(result)
            metrics = result['metrics']
            detection_rows.append({
                'model': model_name,
                'sequence': seq_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'map50': metrics['map50'],
                'map5095': metrics['map5095'],
                'person_map5095': metrics.get('person_map5095'),
                'car_map5095': metrics.get('car_map5095'),
                'summary_path': str(Path(result['output_dir']) / 'summary.json'),
            })
    return {
        'timestamp': datetime.now().isoformat(),
        'rows': detection_rows,
    }


def make_tracking_args(weights: Path, output_dir: Path) -> Namespace:
    config = load_yaml(ROOT / 'configs' / 'tracking_config.yaml')
    detector = config.get('detector', {})
    runtime = config.get('runtime', {})
    trackers = config.get('trackers', {})
    common = trackers.get('common', {})
    bytetrack = trackers.get('bytetrack', {})

    return Namespace(
        config=str(ROOT / 'configs' / 'tracking_config.yaml'),
        weights=str(weights),
        data=str(ROOT / 'data' / 'videos' / 'chapter6_eval'),
        tracker='bytetrack',
        output=str(output_dir),
        save_vid=True,
        save_txt=True,
        show=False,
        overlay=False,
        no_overlay=True,
        conf_thres=detector.get('conf_thres', 0.25),
        nms_thres=detector.get('nms_thres', 0.45),
        img_size=detector.get('img_size', 640),
        half=bool(detector.get('half', False)),
        warmup=bool(detector.get('warmup', True)),
        no_warmup=not bool(detector.get('warmup', True)),
        device=str(runtime.get('device', '0')),
        max_age=common.get('max_age', 30),
        min_hits=common.get('min_hits', 3),
        fps_alpha=runtime.get('fps_alpha', 0.12),
        debug=False,
        track_visible_lag=common.get('visible_lag', 8),
        tracker_iou_thres=bytetrack.get('iou_threshold', 0.3),
        deepsort_max_cosine_distance=0.2,
        deepsort_nn_budget=100,
        bytetrack_high_thres=bytetrack.get('high_threshold', 0.5),
        bytetrack_low_thres=bytetrack.get('low_threshold', 0.1),
        bytetrack_match_thres=bytetrack.get('match_threshold', 0.3),
        bytetrack_second_match_thres=bytetrack.get('second_match_threshold', 0.2),
        centertrack_center_thres=50.0,
        centertrack_pre_thres=0.3,
    )


def run_tracking(output_root: Path) -> dict:
    from scripts.evaluate.eval_tracking import TrackingRunner

    tracking_rows = []
    for model_name, weights in MODELS.items():
        args = make_tracking_args(weights, output_root / model_name)
        runner = TrackingRunner(args)
        runner.run()

        latest_dir = sorted((output_root / model_name).glob('bytetrack_*'))[-1]
        summary_path = latest_dir / 'summary_metrics.json'
        with summary_path.open('r', encoding='utf-8') as handle:
            summary = json.load(handle)

        for row in summary.get('rows', []):
            video_name = row['video_name']
            sequence = 'seq006' if 'seq006' in video_name else 'seq009' if 'seq009' in video_name else video_name
            tracking_rows.append({
                'model': model_name,
                'sequence': sequence,
                'video_name': video_name,
                'frame_count': row['frame_count'],
                'total_detections': row['total_detections'],
                'matched_tracks': row['matched_tracks'],
                'rendered_tracks': row['rendered_tracks'],
                'unique_ids': row['unique_ids'],
                'id_switch_proxy': row['id_switch_proxy'],
                'match_rate': row['match_rate'],
                'avg_fps': row['avg_fps'],
                'summary_path': str(summary_path),
            })

    return {
        'timestamp': datetime.now().isoformat(),
        'rows': tracking_rows,
    }


def main() -> None:
    detection_output = ROOT / 'outputs' / 'detection' / 'chapter6_seq_split'
    tracking_output = ROOT / 'outputs' / 'tracking' / 'chapter6_seq_split'
    summary_output = ROOT / 'outputs' / 'chapter6_seq_split_summary.json'

    detection_output.mkdir(parents=True, exist_ok=True)
    tracking_output.mkdir(parents=True, exist_ok=True)

    payload = {
        'generated_at': datetime.now().isoformat(),
        'detection': run_detection(detection_output),
        'tracking': run_tracking(tracking_output),
    }

    with summary_output.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f'combined summary saved to: {summary_output}')


if __name__ == '__main__':
    main()