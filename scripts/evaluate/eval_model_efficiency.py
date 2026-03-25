#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验模型效率指标批量评估脚本
评估指标: 参数量(M) / 计算量(GFLOPs) / 推理FPS / 推理延迟(ms)

用法示例:
  python scripts/evaluate/eval_model_efficiency.py
  python scripts/evaluate/eval_model_efficiency.py --half --device 0
  python scripts/evaluate/eval_model_efficiency.py --ablation-dir outputs/ablation_study --stage all --output outputs/detection/efficiency_summary_all.csv
"""

import argparse
import csv
import gc
import os
import pathlib
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
YOLO_DIR = ROOT / "yolov5"
for _p in [str(ROOT), str(YOLO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from utils.general import non_max_suppression  # noqa: E402

CANONICAL_ABLATION_ORDER = [
    "ablation_exp01_baseline",
    "ablation_exp02_ghost",
    "ablation_exp03_shuffle",
    "ablation_exp04_attention",
    "ablation_exp05_coordatt",
    "ablation_exp06_siou",
    "ablation_exp07_eiou",
    "ablation_exp08_ghost_attention",
    "ablation_exp09_ghost_eiou",
    "ablation_exp10_attention_eiou",
    "ablation_exp11_shuffle_coordatt",
    "ablation_exp12_shuffle_coordatt_siou",
    "ablation_exp13_shuffle_coordatt_eiou",
]

STAGE_EXPERIMENTS = {
    "stage1": CANONICAL_ABLATION_ORDER[:7],
    "stage2": CANONICAL_ABLATION_ORDER[7:],
    "all": CANONICAL_ABLATION_ORDER,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量评估消融实验效率指标 (Params / GFLOPs / FPS)")
    p.add_argument("--ablation-dir", default="outputs/ablation_study",
                   help="消融实验根目录 (默认 outputs/ablation_study)")
    p.add_argument("--stage", default="all", choices=["stage1", "stage2", "all"],
                   help="实验阶段过滤")
    p.add_argument("--weights-name", default="best.pt",
                   help="权重文件名 (默认 best.pt)")
    p.add_argument("--device", default="0",
                   help="计算设备: 0/1/cpu (默认 0)")
    p.add_argument("--img-size", type=int, default=640,
                   help="推理输入尺寸 (默认 640)")
    p.add_argument("--warmup", type=int, default=50,
                   help="预热迭代次数 (默认 50)")
    p.add_argument("--bench", type=int, default=200,
                   help="正式计时迭代次数 (默认 200)")
    p.add_argument("--half", action="store_true", default=False,
                   help="使用 FP16 推理 (需 CUDA, 默认 FP32)")
    p.add_argument("--output", default=None,
                   help="输出 CSV 路径 (默认 outputs/detection/efficiency_summary_all.csv)")
    return p.parse_args()


def extract_exp_no(name: str) -> int:
    m = re.search(r"exp(\d+)", name)
    return int(m.group(1)) if m else 999


def _load_ckpt_model(weight_path: Path):
    """从 .pt 加载模型，跨平台兼容。返回 float + eval 的模型对象。"""
    orig_posix = pathlib.PosixPath
    if sys.platform.startswith("win"):
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        ckpt = torch.load(str(weight_path), map_location="cpu")
    finally:
        pathlib.PosixPath = orig_posix

    if isinstance(ckpt, dict):
        model = ckpt.get("ema") or ckpt.get("model")
    else:
        model = ckpt

    if model is None or not callable(getattr(model, "forward", None)):
        raise RuntimeError("无法从 checkpoint 提取模型对象")

    model = model.float().eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # 融合 Conv+BN，加速推理并获得更准确的部署性能
    if hasattr(model, "fuse"):
        try:
            model.fuse()
        except Exception:
            pass  # 部分旧版 checkpoint 不支持 fuse，忽略

    return model


def count_params_m(model) -> float:
    """返回参数量（单位 M）。"""
    return sum(p.numel() for p in model.parameters()) / 1_000_000


def compute_gflops(model, imgsz: int) -> float:
    """
    使用 thop 计算 GFLOPs（与 YOLOv5 model_info 口径一致）。
    先在 stride 分辨率下 profile，再按像素面积比例缩放到完整 imgsz。
    失败时返回 -1.0。
    """
    try:
        import thop  # noqa: F401 — 检查 thop 可用性

        stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        first_param = next(model.parameters())
        in_channels = first_param.shape[1]

        # 在 stride 分辨率上 profile（不需要 GPU，降低显存占用）
        dummy = torch.zeros(1, in_channels, stride, stride)
        m_copy = deepcopy(model).cpu().float()
        macs, _ = thop.profile(m_copy, inputs=(dummy,), verbose=False)
        del m_copy, dummy

        # GFLOPs at stride resolution → scale to imgsz
        # 1 MACs = 2 FLOPs；再乘以 (imgsz/stride)^2 面积比
        gflops_stride = macs / 1e9 * 2
        gflops = gflops_stride * (imgsz / stride) ** 2
        return gflops

    except ImportError:
        print("      ⚠ thop 未安装，跳过 GFLOPs 计算 (pip install thop)")
        return -1.0
    except Exception as e:
        print(f"      ⚠ GFLOPs 计算失败: {e}")
        return -1.0


def measure_fps(model, imgsz: int, device: torch.device,
                 warmup: int, bench: int, half: bool):
    """
    使用 CUDA Event（GPU）或 perf_counter（CPU）对推理速度进行基准测试。
    返回 (fps, inf_ms_per_image)，失败时返回 (-1.0, -1.0)。
    """
    try:
        m = deepcopy(model).to(device).eval()
        if half and device.type != "cpu":
            m = m.half()
        else:
            m = m.float()
            half = False

        dtype = torch.float16 if half else torch.float32
        dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=dtype, device=device)

        # 预热（推理 + NMS）
        with torch.no_grad():
            for _ in range(warmup):
                out = m(dummy)
                pred = out[0] if isinstance(out, (tuple, list)) else out
                non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        if device.type != "cpu":
            torch.cuda.synchronize(device)

        # 基准计时（推理 + NMS，不含前处理）
        with torch.no_grad():
            if device.type != "cpu":
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                for _ in range(bench):
                    out = m(dummy)
                    pred = out[0] if isinstance(out, (tuple, list)) else out
                    non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
                end_evt.record()
                torch.cuda.synchronize(device)
                elapsed_ms = start_evt.elapsed_time(end_evt)
            else:
                t0 = time.perf_counter()
                for _ in range(bench):
                    out = m(dummy)
                    pred = out[0] if isinstance(out, (tuple, list)) else out
                    non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

        inf_ms = elapsed_ms / bench
        fps = 1000.0 / inf_ms
        return fps, inf_ms

    except Exception as e:
        print(f"      ⚠ FPS 测量失败: {e}")
        return -1.0, -1.0
    finally:
        try:
            del m
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()


def main() -> None:
    args = parse_args()

    # 解析路径
    ablation_dir = (ROOT / args.ablation_dir
                    if not Path(args.ablation_dir).is_absolute()
                    else Path(args.ablation_dir))
    if not ablation_dir.exists():
        print(f"❌ 消融目录不存在: {ablation_dir}")
        sys.exit(1)

    # 解析设备
    device_str = f"cuda:{args.device}" if args.device.isdigit() else args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
        print("⚠ GPU 不可用，回退到 CPU 评估")
    device = torch.device(device_str)
    print(f"计算设备: {device}")

    precision_tag = "FP16" if (args.half and device.type != "cpu") else "FP32"
    print(f"推理精度: {precision_tag}")
    print(f"图像尺寸: {args.img_size}×{args.img_size}")
    print(f"Bench 迭代: warmup={args.warmup}, bench={args.bench}")

    # 输出路径
    output_path = (Path(args.output) if args.output
                   else ROOT / "outputs" / "detection" / "efficiency_summary_all.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    experiments = STAGE_EXPERIMENTS[args.stage]
    rows: list[dict] = []

    sep = "=" * 75
    print(f"\n{sep}")
    print(f"{'实验名称':<38} {'Params(M)':>9} {'GFLOPs':>8} {'FPS':>8} {'ms/img':>8}")
    print(sep)

    for exp_name in experiments:
        exp_no = extract_exp_no(exp_name)
        weight_path = ablation_dir / exp_name / "weights" / args.weights_name
        row: dict = {
            "exp_name": exp_name,
            "exp_no": exp_no,
            "weights": str(weight_path),
            "params_m": "NA",
            "gflops": "NA",
            "fps": "NA",
            "inf_ms": "NA",
        }

        if not weight_path.exists():
            print(f"  ⚠  [{exp_no:02d}] {exp_name:<34} (权重不存在，已跳过)")
            rows.append(row)
            continue

        print(f"\n  [{exp_no:02d}] {exp_name}")

        # 加载模型
        try:
            model = _load_ckpt_model(weight_path)
        except Exception as e:
            print(f"      ❌ 加载失败: {e}")
            rows.append(row)
            continue

        # 参数量
        params_m = count_params_m(model)
        print(f"      Params : {params_m:.3f} M")

        # GFLOPs
        gflops = compute_gflops(model, args.img_size)
        if gflops >= 0:
            print(f"      GFLOPs : {gflops:.2f} G")

        # FPS / 延迟
        fps, inf_ms = measure_fps(model, args.img_size, device,
                                   args.warmup, args.bench, args.half)
        if fps >= 0:
            print(f"      FPS    : {fps:.1f}  ({inf_ms:.2f} ms/img)")

        row.update({
            "params_m": f"{params_m:.4f}",
            "gflops":   f"{gflops:.4f}" if gflops >= 0 else "NA",
            "fps":      f"{fps:.2f}"    if fps >= 0 else "NA",
            "inf_ms":   f"{inf_ms:.4f}" if inf_ms >= 0 else "NA",
        })
        rows.append(row)

        # 显式释放，避免跨模型 CUDA 残留干扰下一个实验
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # 汇总打印
    print(f"\n{sep}")
    print(f"{'实验名称':<38} {'Params(M)':>9} {'GFLOPs':>8} {'FPS':>8} {'ms/img':>8}")
    print("-" * 75)
    for r in rows:
        print(f"  {r['exp_name']:<36} {r['params_m']:>9} {r['gflops']:>8} {r['fps']:>8} {r['inf_ms']:>8}")
    print(sep)

    # 写出 CSV
    fieldnames = ["exp_name", "exp_no", "weights", "params_m", "gflops", "fps", "inf_ms"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ 效率指标已保存至: {output_path}")
    print(f"   绘图命令:")
    print(f"   python scripts/evaluate/plot_model_efficiency.py --csv {output_path}")


if __name__ == "__main__":
    main()
