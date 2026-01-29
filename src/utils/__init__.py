#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities module providing metrics, visualization, and logging helpers.
"""

from .metrics import (
    compute_iou,
    compute_batch_iou,
    compute_precision_recall,
    compute_ap,
    compute_map,
    MOTMetricsCalculator,
    save_metrics_to_json,
    load_metrics_from_json,
)

from .visualization import (
    generate_color_list,
    get_id_color,
    draw_bounding_box,
    draw_detection_results,
    draw_tracking_results,
    draw_info_panel,
    create_image_grid,
    save_visualization_video,
)

from .logger import (
    LogManager,
    TrainingLogger,
    ProgressBar,
    get_logger,
    init_logger,
    log_debug,
    log_info,
    log_warning,
    log_error,
)

__all__ = [
    # Metrics
    'compute_iou',
    'compute_batch_iou',
    'compute_precision_recall',
    'compute_ap',
    'compute_map',
    'MOTMetricsCalculator',
    'save_metrics_to_json',
    'load_metrics_from_json',
    # Visualization
    'generate_color_list',
    'get_id_color',
    'draw_bounding_box',
    'draw_detection_results',
    'draw_tracking_results',
    'draw_info_panel',
    'create_image_grid',
    'save_visualization_video',
    # Logging
    'LogManager',
    'TrainingLogger',
    'ProgressBar',
    'get_logger',
    'init_logger',
    'log_debug',
    'log_info',
    'log_warning',
    'log_error',
]
