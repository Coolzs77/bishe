"""
可视化工具模块

提供目标检测和多目标跟踪结果的可视化功能，包括绘制边界框、轨迹、信息面板等。
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict


def generate_color_list(num: int) -> List[Tuple[int, int, int]]:
    """
    生成一组区分度高的颜色列表
    
    Args:
        num: 需要生成的颜色数量
    
    Returns:
        颜色列表，每个颜色为 (B, G, R) 格式的元组
    """
    colors = []
    
    # 预定义一些高对比度的颜色
    predefined_colors = [
        (255, 0, 0),      # 蓝色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 红色
        (255, 255, 0),    # 青色
        (255, 0, 255),    # 洋红
        (0, 255, 255),    # 黄色
        (128, 0, 255),    # 橙色
        (255, 128, 0),    # 天蓝色
        (0, 128, 255),    # 橙红色
        (128, 255, 0),    # 草绿色
        (255, 0, 128),    # 紫色
        (0, 255, 128),    # 青绿色
    ]
    
    # 首先使用预定义颜色
    for i in range(min(num, len(predefined_colors))):
        colors.append(predefined_colors[i])
    
    # 如果需要更多颜色，使用HSV空间生成
    if num > len(predefined_colors):
        for i in range(len(predefined_colors), num):
            # 在HSV空间均匀采样色相
            hue = int(180 * (i - len(predefined_colors)) / (num - len(predefined_colors)))
            # 创建HSV颜色
            hsv_color = np.uint8([[[hue, 255, 255]]])
            # 转换为BGR
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in bgr_color))
    
    return colors


def get_id_color(track_id: int, num_colors: int = 256) -> Tuple[int, int, int]:
    """
    根据跟踪ID获取对应的颜色
    
    使用哈希方式确保相同ID始终映射到相同颜色，不影响全局随机状态
    
    Args:
        track_id: 跟踪目标的ID
        num_colors: 颜色空间大小
    
    Returns:
        颜色元组 (B, G, R)
    """
    # 使用独立的随机数生成器，避免影响全局随机状态
    rng = np.random.RandomState(int(track_id) % (2**31))
    hue = rng.randint(0, 180)
    saturation = rng.randint(150, 256)
    value = rng.randint(150, 256)
    
    # 转换HSV到BGR
    hsv_color = np.uint8([[[hue, saturation, value]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    
    return tuple(int(c) for c in bgr_color)


def draw_bounding_box(
    image: np.ndarray,
    bbox: Union[List, Tuple, np.ndarray],
    class_name: Optional[str] = None,
    confidence: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制单个边界框
    
    Args:
        image: 输入图像 (BGR格式)
        bbox: 边界框坐标 [x1, y1, x2, y2]
        class_name: 类别名称，可选
        confidence: 置信度，可选
        color: 边界框颜色 (B, G, R)
        thickness: 线条粗细
    
    Returns:
        绘制后的图像
    """
    # 复制图像以避免修改原图
    img = image.copy()
    
    # 获取边界框坐标
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # 绘制边界框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # 准备标签文本
    if class_name is not None or confidence is not None:
        label_parts = []
        if class_name is not None:
            label_parts.append(class_name)
        if confidence is not None:
            label_parts.append(f'{confidence:.2f}')
        label = ' '.join(label_parts)
        
        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # 绘制标签背景
        label_y1 = max(y1 - text_height - 10, 0)
        label_y2 = y1
        cv2.rectangle(img, (x1, label_y1), (x1 + text_width + 4, label_y2), color, -1)
        
        # 绘制标签文本（白色）
        cv2.putText(
            img, label, (x1 + 2, label_y2 - 4),
            font, font_scale, (255, 255, 255), font_thickness
        )
    
    return img


def draw_detection_results(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: Optional[List[str]] = None,
    confidences: Optional[np.ndarray] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制检测结果
    
    Args:
        image: 输入图像 (BGR格式)
        boxes: 边界框数组，形状为 (N, 4)，格式为 [x1, y1, x2, y2]
        classes: 类别名称列表，长度为 N
        confidences: 置信度数组，形状为 (N,)
        colors: 颜色列表，如果为None则自动生成
        thickness: 线条粗细
    
    Returns:
        绘制后的图像
    """
    img = image.copy()
    
    if len(boxes) == 0:
        return img
    
    boxes = np.asarray(boxes)
    num_boxes = len(boxes)
    
    # 生成颜色
    if colors is None:
        if classes is not None:
            # 为每个类别分配固定颜色
            unique_classes = list(set(classes))
            class_colors = generate_color_list(len(unique_classes))
            class_to_color = {cls: color for cls, color in zip(unique_classes, class_colors)}
            colors = [class_to_color[cls] for cls in classes]
        else:
            colors = [(0, 255, 0)] * num_boxes
    
    # 绘制每个边界框
    for i in range(num_boxes):
        class_name = classes[i] if classes is not None else None
        conf = confidences[i] if confidences is not None else None
        color = colors[i] if i < len(colors) else (0, 255, 0)
        
        img = draw_bounding_box(img, boxes[i], class_name, conf, color, thickness)
    
    return img


# 用于存储轨迹历史的全局字典
_trajectory_history = defaultdict(list)


def draw_tracking_results(
    image: np.ndarray,
    boxes: np.ndarray,
    track_ids: np.ndarray,
    classes: Optional[List[str]] = None,
    draw_trajectory: bool = True,
    trajectory_history: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    max_trajectory_length: int = 30,
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制跟踪结果
    
    Args:
        image: 输入图像 (BGR格式)
        boxes: 边界框数组，形状为 (N, 4)
        track_ids: 跟踪ID数组，形状为 (N,)
        classes: 类别名称列表，长度为 N
        draw_trajectory: 是否绘制运动轨迹
        trajectory_history: 轨迹历史字典，格式为 {track_id: [(x, y), ...]}
                          如果为None则使用全局历史
        max_trajectory_length: 轨迹最大长度
        thickness: 线条粗细
    
    Returns:
        绘制后的图像
    """
    global _trajectory_history
    
    img = image.copy()
    
    if len(boxes) == 0:
        return img
    
    boxes = np.asarray(boxes)
    track_ids = np.asarray(track_ids)
    
    # 使用外部轨迹历史或全局历史
    if trajectory_history is None:
        trajectory_history = _trajectory_history
    
    # 绘制每个跟踪目标
    for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
        track_id = int(track_id)
        color = get_id_color(track_id)
        
        # 绘制边界框
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # 准备标签
        label_parts = [f'ID:{track_id}']
        if classes is not None and i < len(classes):
            label_parts.insert(0, classes[i])
        label = ' '.join(label_parts)
        
        # 绘制标签背景和文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        label_y1 = max(y1 - text_height - 10, 0)
        cv2.rectangle(img, (x1, label_y1), (x1 + text_width + 4, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 2, y1 - 4),
            font, font_scale, (255, 255, 255), font_thickness
        )
        
        # 更新和绘制轨迹
        if draw_trajectory:
            # 计算边界框中心点
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # 更新轨迹历史
            trajectory_history[track_id].append((center_x, center_y))
            
            # 限制轨迹长度
            if len(trajectory_history[track_id]) > max_trajectory_length:
                trajectory_history[track_id] = trajectory_history[track_id][-max_trajectory_length:]
            
            # 绘制轨迹
            points = trajectory_history[track_id]
            for j in range(1, len(points)):
                # 轨迹颜色渐变（越老越暗）
                alpha = j / len(points)
                line_color = tuple(int(c * alpha) for c in color)
                cv2.line(img, points[j-1], points[j], line_color, max(1, thickness - 1))
    
    return img


def draw_info_panel(
    image: np.ndarray,
    info_dict: Dict[str, Union[str, int, float]],
    position: str = 'top-left',
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.7
) -> np.ndarray:
    """
    在图像上绘制信息面板
    
    Args:
        image: 输入图像 (BGR格式)
        info_dict: 要显示的信息字典
        position: 面板位置，可选 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        bg_color: 背景颜色 (B, G, R)
        text_color: 文本颜色 (B, G, R)
        alpha: 背景透明度
    
    Returns:
        绘制后的图像
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # 准备文本行
    lines = []
    for key, value in info_dict.items():
        if isinstance(value, float):
            lines.append(f'{key}: {value:.4f}')
        else:
            lines.append(f'{key}: {value}')
    
    if not lines:
        return img
    
    # 计算面板大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 20
    padding = 10
    
    max_width = 0
    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        max_width = max(max_width, text_width)
    
    panel_width = max_width + 2 * padding
    panel_height = len(lines) * line_height + 2 * padding
    
    # 确定面板位置
    if 'left' in position:
        x1 = padding
    else:
        x1 = w - panel_width - padding
    
    if 'top' in position:
        y1 = padding
    else:
        y1 = h - panel_height - padding
    
    x2 = x1 + panel_width
    y2 = y1 + panel_height
    
    # 绘制半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 绘制文本
    for i, line in enumerate(lines):
        text_y = y1 + padding + (i + 1) * line_height - 5
        cv2.putText(img, line, (x1 + padding, text_y), font, font_scale, text_color, font_thickness)
    
    return img


def create_image_grid(
    images: List[np.ndarray],
    rows: int,
    cols: int,
    cell_size: Optional[Tuple[int, int]] = None,
    padding: int = 2,
    bg_color: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """
    创建图像网格
    
    Args:
        images: 图像列表
        rows: 网格行数
        cols: 网格列数
        cell_size: 每个单元格的大小 (width, height)，如果为None则使用第一张图像的大小
        padding: 单元格之间的间距
        bg_color: 背景颜色 (B, G, R)
    
    Returns:
        拼接后的网格图像
    """
    if len(images) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 确定单元格大小
    if cell_size is None:
        cell_width = images[0].shape[1]
        cell_height = images[0].shape[0]
    else:
        cell_width, cell_height = cell_size
    
    # 计算网格总大小
    grid_width = cols * cell_width + (cols + 1) * padding
    grid_height = rows * cell_height + (rows + 1) * padding
    
    # 创建背景
    grid = np.full((grid_height, grid_width, 3), bg_color, dtype=np.uint8)
    
    # 填充图像
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        
        row = idx // cols
        col = idx % cols
        
        # 调整图像大小
        if img.shape[0] != cell_height or img.shape[1] != cell_width:
            img = cv2.resize(img, (cell_width, cell_height))
        
        # 确保是3通道
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 计算位置
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)
        
        # 放置图像
        grid[y:y+cell_height, x:x+cell_width] = img
    
    return grid


def save_visualization_video(
    output_path: str,
    frames: List[np.ndarray],
    fps: float = 30.0,
    codec: str = 'mp4v'
) -> bool:
    """
    将帧序列保存为视频文件
    
    Args:
        output_path: 输出视频文件路径
        frames: 帧列表，每帧为BGR格式的numpy数组
        fps: 帧率
        codec: 视频编码器，默认为 'mp4v'
    
    Returns:
        是否保存成功
    """
    if len(frames) == 0:
        import logging
        logging.warning("帧列表为空，无法保存视频")
        return False
    
    # 获取帧大小
    height, width = frames[0].shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        import logging
        logging.error(f"无法创建视频写入器，路径: {output_path}")
        return False
    
    try:
        for frame in frames:
            # 确保帧大小一致
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            
            # 确保是3通道
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            writer.write(frame)
        
        writer.release()
        import logging
        logging.info(f"视频已保存: {output_path} ({len(frames)} 帧, {fps} FPS)")
        return True
    
    except Exception as e:
        import logging
        logging.error(f"保存视频时出错: {e}")
        writer.release()
        return False


def clear_trajectory_history():
    """
    清除全局轨迹历史
    """
    global _trajectory_history
    _trajectory_history.clear()


def get_trajectory_history() -> Dict[int, List[Tuple[int, int]]]:
    """
    获取全局轨迹历史的副本
    
    Returns:
        轨迹历史字典
    """
    global _trajectory_history
    return dict(_trajectory_history)
