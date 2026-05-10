from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'docs' / 'figures'
TMP_DIR = OUT_DIR / '_tmp_thesis_frames'


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc'),
        Path('C:/Windows/Fonts/simhei.ttf'),
        Path('C:/Windows/Fonts/simsun.ttc'),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONT_LABEL = load_font(26, bold=False)
FONT_FLOW_TITLE = load_font(30, bold=True)
FONT_FLOW_BOX = load_font(22, bold=False)
FONT_CALLOUT = load_font(24, bold=True)
FONT_GRID_TITLE = load_font(30, bold=True)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def read_image(path: Path) -> Image.Image:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f'无法读取图像: {path}')
    return bgr_to_pil(image)


def extract_frame(video_path: Path, frame_index: int, cache_name: str) -> Image.Image:
    cache_path = TMP_DIR / cache_name
    if cache_path.exists():
        return Image.open(cache_path).convert('RGB')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {video_path}')

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f'无法从视频提取第 {frame_index} 帧: {video_path}')

    image = bgr_to_pil(frame)
    image.save(cache_path)
    return image


def draw_panel(
    canvas: Image.Image,
    panel: Image.Image,
    label: str,
    x: int,
    y: int,
    cell_w: int,
    cell_h: int,
    label_h: int = 44,
) -> None:
    draw = ImageDraw.Draw(canvas)
    image_box_h = cell_h - label_h
    panel_ratio = panel.width / panel.height
    box_ratio = cell_w / image_box_h

    if panel_ratio >= box_ratio:
        draw_w = cell_w
        draw_h = int(cell_w / panel_ratio)
    else:
        draw_h = image_box_h
        draw_w = int(image_box_h * panel_ratio)

    offset_x = x + (cell_w - draw_w) // 2
    offset_y = y + (image_box_h - draw_h) // 2
    resized = panel.resize((draw_w, draw_h), Image.Resampling.LANCZOS)
    canvas.paste(resized, (offset_x, offset_y))
    draw.rounded_rectangle((x, y, x + cell_w, y + image_box_h), radius=10, outline=(70, 70, 70), width=2)
    draw.rectangle((x, y + image_box_h, x + cell_w, y + cell_h), fill=(248, 248, 248))
    draw.text((x + 10, y + image_box_h + 6), label, fill=(25, 25, 25), font=FONT_LABEL)


def make_grid(
    images: Iterable[Image.Image],
    labels: Iterable[str],
    out_path: Path,
    columns: int,
    cell_w: int,
    cell_h: int,
    title: str | None = None,
    margin: int = 18,
    gap: int = 16,
) -> None:
    image_list = list(images)
    label_list = list(labels)
    rows = (len(image_list) + columns - 1) // columns
    title_h = 54 if title else 0
    canvas_w = margin * 2 + columns * cell_w + (columns - 1) * gap
    canvas_h = margin * 2 + title_h + rows * cell_h + (rows - 1) * gap
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    top_y = margin
    if title:
        title_box = draw.textbbox((0, 0), title, font=FONT_GRID_TITLE)
        title_w = title_box[2] - title_box[0]
        draw.text(((canvas_w - title_w) // 2, margin), title, fill=(28, 28, 28), font=FONT_GRID_TITLE)
        top_y += title_h

    for index, image in enumerate(image_list):
        row = index // columns
        col = index % columns
        x = margin + col * (cell_w + gap)
        y = top_y + row * (cell_h + gap)
        draw_panel(canvas, image, label_list[index], x, y, cell_w, cell_h)

    canvas.save(out_path)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color=(54, 76, 99), width: int = 4) -> None:
    draw.line((start, end), fill=color, width=width)
    end_x, end_y = end
    if start[0] == end_x:
        points = [(end_x, end_y), (end_x - 8, end_y - 14), (end_x + 8, end_y - 14)]
    else:
        points = [(end_x, end_y), (end_x - 14, end_y - 8), (end_x - 14, end_y + 8)]
    draw.polygon(points, fill=color)


def draw_multiline_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
    fill=(24, 24, 24),
    line_gap: int = 6,
) -> None:
    x1, y1, x2, y2 = box
    widths = []
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + line_gap * max(len(lines) - 1, 0)
    cursor_y = y1 + (y2 - y1 - total_h) // 2
    for line, width, height in zip(lines, widths, heights):
        cursor_x = x1 + (x2 - x1 - width) // 2
        draw.text((cursor_x, cursor_y), line, fill=fill, font=font)
        cursor_y += height + line_gap


def make_flowchart(out_path: Path) -> None:
    canvas = Image.new('RGB', (1560, 820), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((46, 30), '图6-1 综合实验与系统演示流程图', fill=(28, 28, 28), font=FONT_FLOW_TITLE)

    boxes = [
        ((60, 120, 340, 230), (235, 244, 255), (69, 123, 157), ['FLIR热视频输入', 'seq006 与 seq009']),
        ((400, 120, 700, 230), (238, 248, 241), (76, 175, 80), ['PC端浮点综合实验', '检测与跟踪方案筛选']),
        ((760, 120, 1060, 230), (255, 245, 232), (230, 145, 56), ['量化与部署验证', 'min-max 与 KL 对比']),
        ((1120, 120, 1450, 230), (242, 238, 255), (132, 94, 194), ['RV1126B板端演示', '系统实时性分析']),
        ((280, 410, 620, 540), (255, 245, 232), (230, 145, 56), ['典型场景效果图', '正常场景与远距场景']),
        ((700, 410, 1040, 540), (238, 248, 241), (76, 175, 80), ['典型失效案例', '漏检、遮挡与ID切换']),
        ((500, 650, 980, 760), (235, 244, 255), (69, 123, 157), ['形成系统级结论', '支撑第7章总结与展望']),
    ]

    for rect, fill, outline, lines in boxes:
        draw.rounded_rectangle(rect, radius=18, fill=fill, outline=outline, width=3)
        draw_multiline_center(draw, rect, lines, FONT_FLOW_BOX)

    draw_arrow(draw, (340, 175), (400, 175))
    draw_arrow(draw, (700, 175), (760, 175))
    draw_arrow(draw, (1060, 175), (1120, 175))
    draw_arrow(draw, (550, 230), (550, 410))
    draw_arrow(draw, (890, 230), (890, 410))
    draw_arrow(draw, (620, 520), (680, 650))
    draw_arrow(draw, (1040, 520), (900, 650))
    canvas.save(out_path)


def add_callout(image: Image.Image, rect: tuple[int, int, int, int], text: str) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rounded_rectangle(rect, radius=10, outline=(210, 35, 35), width=5)
    text_box = draw.textbbox((0, 0), text, font=FONT_CALLOUT)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    tx = rect[0]
    ty = max(12, rect[1] - text_h - 18)
    draw.rounded_rectangle((tx, ty, tx + text_w + 20, ty + text_h + 12), radius=10, fill=(210, 35, 35))
    draw.text((tx + 10, ty + 4), text, fill=(255, 255, 255), font=FONT_CALLOUT)
    return annotated


def latest_result_video(base_dir: Path, video_stem: str) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = sorted(base_dir.glob('bytetrack_*'), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        video_path = candidate / video_stem / 'result.mp4'
        if video_path.exists():
            return video_path
    return None


def build_figures() -> None:
    ensure_dirs()

    board_videos = {
        'baseline_minmax': ROOT / 'outputs' / 'rv1126b_board_results' / 'out_best_baseline_ZAtDSNuZZjkZFvMAo_seq006.mp4',
        'baseline_kl': ROOT / 'outputs' / 'rv1126b_board_results' / 'out_best_baseline_kl_ZAtDSNuZZjkZFvMAo_seq006.mp4',
        'baseline_seq009_kl': ROOT / 'outputs' / 'rv1126b_board_results' / 'out_best_baseline_kl_t3f7QC8hZr6zYXpEZ_seq009.mp4',
        'eiou_minmax': ROOT / 'outputs' / 'rv1126b_board_results' / 'out_best_eiou_ZAtDSNuZZjkZFvMAo_seq006.mp4',
        'eiou_kl': ROOT / 'outputs' / 'rv1126b_board_results' / 'out_best_eiou_kl_ZAtDSNuZZjkZFvMAo_seq006.mp4',
        'ghost_minmax': ROOT / 'outputs' / 'rv1126b_board_results' / 'out_best_ghost_eiou_ZAtDSNuZZjkZFvMAo_seq006.mp4',
        'ghost_kl': ROOT / 'outputs' / 'rv1126b_board_results' / 'out_best_ghost_eiou_kl_ZAtDSNuZZjkZFvMAo_seq006.mp4',
        'ghost_seq006_kl': ROOT / 'outputs' / 'rv1126b_board_results' / 'out_best_ghost_eiou_kl_ZAtDSNuZZjkZFvMAo_seq006.mp4',
        'ghost_seq009_kl': ROOT / 'outputs' / 'rv1126b_board_results' / 'out_best_ghost_eiou_kl_t3f7QC8hZr6zYXpEZ_seq009.mp4',
        'eiou_seq009_kl': ROOT / 'outputs' / 'rv1126b_board_results' / 'out_best_eiou_kl_t3f7QC8hZr6zYXpEZ_seq009.mp4',
    }

    fig53_images = [
        extract_frame(board_videos['baseline_minmax'], 120, 'fig5_3_a_baseline_minmax_seq006_f120.png'),
        extract_frame(board_videos['baseline_kl'], 120, 'fig5_3_b_baseline_kl_seq006_f120.png'),
        extract_frame(board_videos['eiou_minmax'], 120, 'fig5_3_c_eiou_minmax_seq006_f120.png'),
        extract_frame(board_videos['eiou_kl'], 120, 'fig5_3_d_eiou_kl_seq006_f120.png'),
        extract_frame(board_videos['ghost_minmax'], 120, 'fig5_3_e_ghost_minmax_seq006_f120.png'),
        extract_frame(board_videos['ghost_kl'], 120, 'fig5_3_f_ghost_kl_seq006_f120.png'),
    ]
    fig53_labels = [
        '（a）Baseline，min-max量化',
        '（b）Baseline，KL量化',
        '（c）EIoU，min-max量化',
        '（d）EIoU，KL量化',
        '（e）Ghost+EIoU，min-max量化',
        '（f）Ghost+EIoU，KL量化',
    ]
    make_grid(
        fig53_images,
        fig53_labels,
        OUT_DIR / 'fig5-3.png',
        columns=2,
        cell_w=760,
        cell_h=430,
        title='FLIR视频 seq006 第120帧板端输出对比',
    )

    fig54_images = [
        read_image(ROOT / 'outputs' / 'rv1126b_board_results' / 'outputs' / 'test_00_out.png'),
        read_image(ROOT / 'outputs' / 'rv1126b_board_results' / 'outputs' / 'test_01_out.png'),
        read_image(ROOT / 'outputs' / 'rv1126b_board_results' / 'outputs' / 'test_02_out.png'),
        read_image(ROOT / 'outputs' / 'rv1126b_board_results' / 'outputs' / 'test_03_out.png'),
    ]
    fig54_labels = [
        '（a）静态测试图像1检测结果',
        '（b）静态测试图像2检测结果',
        '（c）静态测试图像3检测结果',
        '（d）静态测试图像4检测结果',
    ]
    make_grid(fig54_images, fig54_labels, OUT_DIR / 'fig5-4.png', columns=2, cell_w=760, cell_h=575)

    make_flowchart(OUT_DIR / 'fig6-1.png')

    pc_dir = ROOT / 'outputs' / 'tracking' / 'chapter6_seq_split_overlay'
    fig62_video_paths = {
        'baseline_seq006': latest_result_video(pc_dir / 'baseline', 'ZAtDSNuZZjkZFvMAo_seq006'),
        'baseline_seq009': latest_result_video(pc_dir / 'baseline', 't3f7QC8hZr6zYXpEZ_seq009'),
        'eiou_seq006': latest_result_video(pc_dir / 'eiou', 'ZAtDSNuZZjkZFvMAo_seq006'),
        'eiou_seq009': latest_result_video(pc_dir / 'eiou', 't3f7QC8hZr6zYXpEZ_seq009'),
        'ghost_seq006': latest_result_video(pc_dir / 'ghost_eiou', 'ZAtDSNuZZjkZFvMAo_seq006'),
        'ghost_seq009': latest_result_video(pc_dir / 'ghost_eiou', 't3f7QC8hZr6zYXpEZ_seq009'),
    }

    if all(path is not None and path.exists() for path in fig62_video_paths.values()):
        fig62_images = [
            extract_frame(fig62_video_paths['baseline_seq006'], 120, 'fig6_2_v3_a_baseline_pc_seq006_f120.png'),
            extract_frame(fig62_video_paths['baseline_seq009'], 300, 'fig6_2_v3_b_baseline_pc_seq009_f300.png'),
            extract_frame(fig62_video_paths['eiou_seq006'], 120, 'fig6_2_v3_c_eiou_pc_seq006_f120.png'),
            extract_frame(fig62_video_paths['eiou_seq009'], 300, 'fig6_2_v3_d_eiou_pc_seq009_f300.png'),
            extract_frame(fig62_video_paths['ghost_seq006'], 120, 'fig6_2_v3_e_ghost_pc_seq006_f120.png'),
            extract_frame(fig62_video_paths['ghost_seq009'], 300, 'fig6_2_v3_f_ghost_pc_seq009_f300.png'),
        ]
    else:
        fig62_images = [
            read_image(ROOT / 'outputs' / 'tracking' / 'chapter4_frames' / 'bytetrack_seq006_f120.jpg'),
            read_image(ROOT / 'outputs' / 'tracking' / 'chapter4_frames' / 'bytetrack_seq009_f300.jpg'),
            read_image(ROOT / 'outputs' / 'tracking' / 'chapter4_frames' / 'bytetrack_seq006_f120.jpg'),
            read_image(ROOT / 'outputs' / 'tracking' / 'chapter4_frames' / 'bytetrack_seq009_f300.jpg'),
            read_image(ROOT / 'outputs' / 'tracking' / 'chapter4_frames' / 'bytetrack_seq006_f120.jpg'),
            read_image(ROOT / 'outputs' / 'tracking' / 'chapter4_frames' / 'bytetrack_seq009_f300.jpg'),
        ]
    fig62_labels = [
        '（a）Baseline+ByteTrack，seq006第120帧',
        '（b）Baseline+ByteTrack，seq009第300帧',
        '（c）EIoU+ByteTrack，seq006第120帧',
        '（d）EIoU+ByteTrack，seq009第300帧',
        '（e）Ghost+EIoU+ByteTrack，seq006第120帧',
        '（f）Ghost+EIoU+ByteTrack，seq009第300帧',
    ]
    make_grid(fig62_images, fig62_labels, OUT_DIR / 'fig6-2.png', columns=2, cell_w=760, cell_h=575)

    fig63_images = [
        extract_frame(board_videos['baseline_kl'], 120, 'fig6_3_v2_a_baseline_board_seq006_f120.png'),
        extract_frame(board_videos['baseline_seq009_kl'], 300, 'fig6_3_v2_b_baseline_board_seq009_f300.png'),
        extract_frame(board_videos['eiou_kl'], 120, 'fig6_3_v2_c_eiou_board_seq006_f120.png'),
        extract_frame(board_videos['eiou_seq009_kl'], 300, 'fig6_3_v2_d_eiou_board_seq009_f300.png'),
        extract_frame(board_videos['ghost_seq006_kl'], 120, 'fig6_3_v2_e_ghost_board_seq006_f120.png'),
        extract_frame(board_videos['ghost_seq009_kl'], 300, 'fig6_3_v2_f_ghost_board_seq009_f300.png'),
    ]
    fig63_labels = [
        '（a）Baseline板端结果，seq006第120帧',
        '（b）Baseline板端结果，seq009第300帧',
        '（c）EIoU板端结果，seq006第120帧',
        '（d）EIoU板端结果，seq009第300帧',
        '（e）Ghost+EIoU板端结果，seq006第120帧',
        '（f）Ghost+EIoU板端结果，seq009第300帧',
    ]
    make_grid(fig63_images, fig63_labels, OUT_DIR / 'fig6-3.png', columns=2, cell_w=760, cell_h=575)

    if all(path is not None and path.exists() for path in fig62_video_paths.values()):
        miss_frame = extract_frame(fig62_video_paths['ghost_seq009'], 420, 'fig6_4_v4_a_scene_pc_seq009_f420.png')
        switch_frame = extract_frame(fig62_video_paths['eiou_seq006'], 185, 'fig6_4_v4_b_scene_pc_seq006_f185.png')
    else:
        miss_frame = read_image(ROOT / 'outputs' / 'tracking' / 'chapter4_frames' / 'bytetrack_seq009_f300.jpg')
        switch_frame = read_image(ROOT / 'outputs' / 'tracking' / 'chapter4_frames' / 'bytetrack_seq006_f120.jpg')

    fig64_images = [
        miss_frame,
        switch_frame,
    ]
    fig64_labels = [
        '（a）Ghost+EIoU，seq009第420帧典型场景',
        '（b）EIoU，seq006第185帧典型场景',
    ]
    make_grid(fig64_images, fig64_labels, OUT_DIR / 'fig6-4.png', columns=2, cell_w=760, cell_h=575)

    print('Generated thesis figures: fig5-3, fig5-4, fig6-1, fig6-2, fig6-3, fig6-4')


if __name__ == '__main__':
    build_figures()