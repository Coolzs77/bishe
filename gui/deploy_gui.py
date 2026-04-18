# -*- coding: utf-8 -*-
"""
红外多目标检测与跟踪系统 — 统一工作台 (PyQt5)

全流程模块：数据处理 / 模型训练 / 检测评估 / 跟踪评估 / 板端部署
设计风格：Art Deco 亮色，大字体固定尺寸
所有页面统一 Splitter (左参数 + 右预览) 布局

启动:  python gui/deploy_gui.py
依赖:  pip install PyQt5 paramiko opencv-python
"""

import sys, os, re, subprocess, threading, json, glob, random, csv
from datetime import datetime
from pathlib import Path

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import numpy as np
except ImportError:
    np = None

try:
    import paramiko
    _PARAMIKO = True
except ImportError:
    _PARAMIKO = False

from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QTimer, QSize)
from PyQt5.QtGui import (QFont, QImage, QPixmap, QColor)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QLineEdit,
    QTextEdit, QGroupBox, QTabWidget, QSplitter, QSlider,
    QCheckBox, QSpinBox, QDoubleSpinBox, QFileDialog,
    QMessageBox, QSizePolicy, QStatusBar, QFrame, QProgressBar
)

# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent.parent

UBUNTU_HOST = "192.168.11.128"
UBUNTU_USER = "coolzs77"
UBUNTU_PASS = "0221"
BOARD_DEPLOY = "/userdata/bishe_rknn_yolov5"
ADB_SERIAL = "4fce67e85c12d1a7"

ABLATION_DIR = ROOT / "outputs" / "ablation_study"
BOARD_RESULTS_DIR = ROOT / "outputs" / "rv1126b_board_results"
TRACKING_RESULTS_DIR = ROOT / "outputs" / "tracking"
DETECTION_RESULTS_DIR = ROOT / "outputs" / "detection"
RESULTS_DIR = ROOT / "outputs" / "results"

TRACKERS = ["deepsort", "bytetrack", "centertrack"]

BOARD_MODELS = [
    ("EIoU (normal)", "best_eiou.rknn"),
    ("Baseline (normal)", "best_baseline.rknn"),
    ("Ghost+EIoU (normal)", "best_ghost_eiou.rknn"),
    ("EIoU (kl)", "best_eiou_kl.rknn"),
    ("Baseline (kl)", "best_baseline_kl.rknn"),
    ("Ghost+EIoU (kl)", "best_ghost_eiou_kl.rknn"),
]

BOARD_VIDEOS = [
    ("seq009 — 565 帧", "t3f7QC8hZr6zYXpEZ_seq009.mp4"),
    ("seq006 — 221 帧", "ZAtDSNuZZjkZFvMAo_seq006.mp4"),
]

BOARD_IMAGES = [
    ("test_00", "test_00.jpg"),
    ("test_01", "test_01.jpg"),
    ("test_02", "test_02.jpg"),
    ("test_03", "test_03.jpg"),
    ("test_04", "test_04.jpg"),
]

# ── 配色方案 (Art Deco 亮色) ──
C_BG        = "#f4f5f7"
C_PANEL     = "#ffffff"
C_PANEL_ALT = "#f8f9fb"
C_BORDER    = "#d1d5db"
C_BORDER_L  = "#e5e7eb"
C_ACCENT    = "#1e3a5f"
C_ACCENT_L  = "#2563eb"
C_GOLD      = "#b8860b"
C_GOLD_L    = "#d4a843"
C_TEXT      = "#1f2937"
C_TEXT_S    = "#6b7280"
C_OK        = "#16a34a"
C_ERR       = "#dc2626"
C_WARN      = "#d97706"
C_TAB_BG    = "#e8ebf0"
C_HEADER_L  = "#1e3a5f"
C_HEADER_R  = "#2563eb"


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════
def find_weights():
    """扫描消融目录中的 best.pt 权重"""
    results = []
    if ABLATION_DIR.is_dir():
        for exp in sorted(ABLATION_DIR.iterdir()):
            pt = exp / "weights" / "best.pt"
            if pt.exists():
                results.append((exp.name, str(pt)))
    # 也扫描 ablation_old
    old = ROOT / "outputs" / "ablation_old"
    if old.is_dir():
        for exp in sorted(old.iterdir()):
            pt = exp / "weights" / "best.pt"
            if pt.exists():
                results.append((f"(old) {exp.name}", str(pt)))
    return results


def find_files(dirs, pattern):
    """在多个目录中递归搜索匹配的文件"""
    items = []
    for d in dirs:
        if d.is_dir():
            for f in sorted(d.rglob(pattern)):
                try:
                    rel = f.relative_to(ROOT)
                    items.append((str(rel), str(f)))
                except ValueError:
                    items.append((f.name, str(f)))
    return items


def path_valid(p, kind="file"):
    """检查路径是否存在"""
    pp = Path(p) if Path(p).is_absolute() else ROOT / p
    return pp.is_file() if kind == "file" else pp.is_dir()


# ═══════════════════════════════════════════════════════════════
# 后台线程
# ═══════════════════════════════════════════════════════════════
class LocalCmdWorker(QThread):
    log_line = pyqtSignal(str, str)
    finished = pyqtSignal(int)

    def __init__(self, cmd, cwd=None, parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self.cwd = cwd or str(ROOT)

    def run(self):
        try:
            self.log_line.emit(f"❯ {self.cmd}", "cmd")
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.Popen(
                self.cmd, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
                cwd=self.cwd, encoding="utf-8", errors="replace",
                env=env)
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip("\n\r")
                if line:
                    self.log_line.emit(line, "info")
            proc.wait()
            code = proc.returncode
            if code == 0:
                self.log_line.emit("━━ 完成 (exit 0) ━━", "ok")
            else:
                self.log_line.emit(f"━━ 失败 (exit {code}) ━━", "err")
            self.finished.emit(code)
        except Exception as e:
            self.log_line.emit(f"━━ 异常: {e} ━━", "err")
            self.finished.emit(-1)


class SSHWorker(QThread):
    """通过 SSH 在 Ubuntu 上执行命令"""
    log_line = pyqtSignal(str, str)
    finished = pyqtSignal(int)
    stdout_text = pyqtSignal(str)   # 完整输出

    def __init__(self, cmd, parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self._all_out = []

    def run(self):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(UBUNTU_HOST, username=UBUNTU_USER,
                        password=UBUNTU_PASS, timeout=10)
            self.log_line.emit(f"[SSH] 已连接 {UBUNTU_HOST}", "ok")
            self.log_line.emit(f"❯ {self.cmd}", "cmd")
            _, stdout, stderr = ssh.exec_command(
                self.cmd, get_pty=True, timeout=600)
            for raw in iter(stdout.readline, ""):
                line = raw.rstrip("\n\r")
                if not line:
                    continue
                self._all_out.append(line)
                lvl = "info"
                ll = line.lower()
                if "error" in ll or "fail" in ll:
                    lvl = "err"
                elif "done" in ll or "完成" in line or "PULL_OK" in line:
                    lvl = "ok"
                self.log_line.emit(line, lvl)
            err = stderr.read().decode(errors="replace").strip()
            if err:
                for el in err.splitlines():
                    self._all_out.append(el)
                    self.log_line.emit(el, "err")
            ssh.close()
            full = "\n".join(self._all_out)
            # 检查是否有 adb error
            if "error" in full.lower() and "No such file" in full:
                self.log_line.emit(
                    "━━ 操作失败: 远程文件不存在 ━━", "err")
                self.finished.emit(1)
            else:
                self.log_line.emit("━━ SSH 会话结束 ━━", "ok")
                self.finished.emit(0)
            self.stdout_text.emit(full)
        except paramiko.AuthenticationException:
            self.log_line.emit(
                "━━ SSH 认证失败: 检查用户名/密码 ━━", "err")
            self.finished.emit(-1)
        except paramiko.SSHException as e:
            self.log_line.emit(f"━━ SSH 连接异常: {e} ━━", "err")
            self.finished.emit(-1)
        except TimeoutError:
            self.log_line.emit("━━ SSH 连接超时 ━━", "err")
            self.finished.emit(-1)
        except Exception as e:
            self.log_line.emit(f"━━ SSH 错误: {e} ━━", "err")
            self.finished.emit(-1)


class SFTPWorker(QThread):
    """从 Ubuntu SFTP 下载文件到 Windows 本地"""
    log_line = pyqtSignal(str, str)
    finished = pyqtSignal(int, str)  # (code, local_path)

    def __init__(self, remote_path, local_path, parent=None):
        super().__init__(parent)
        self.remote = remote_path
        self.local = local_path

    def run(self):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(UBUNTU_HOST, username=UBUNTU_USER,
                        password=UBUNTU_PASS, timeout=10)
            sftp = ssh.open_sftp()
            # 验证远程文件存在
            try:
                sftp.stat(self.remote)
            except FileNotFoundError:
                self.log_line.emit(
                    f"━━ Ubuntu 上不存在: {self.remote} ━━", "err")
                sftp.close()
                ssh.close()
                self.finished.emit(1, "")
                return
            os.makedirs(os.path.dirname(self.local), exist_ok=True)
            self.log_line.emit(
                f"⬇ 下载: {self.remote} → {self.local}", "info")
            sftp.get(self.remote, self.local)
            sftp.close()
            ssh.close()
            self.log_line.emit(f"━━ 已保存: {self.local} ━━", "ok")
            self.finished.emit(0, self.local)
        except Exception as e:
            self.log_line.emit(f"━━ SFTP 失败: {e} ━━", "err")
            self.finished.emit(-1, "")


class SFTPUploadWorker(QThread):
    """从 Windows 本地 SFTP 上传文件到 Ubuntu"""
    log_line = pyqtSignal(str, str)
    finished = pyqtSignal(int, str)  # (code, remote_path)

    def __init__(self, local_path, remote_path, parent=None):
        super().__init__(parent)
        self.local = local_path
        self.remote = remote_path

    def run(self):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(UBUNTU_HOST, username=UBUNTU_USER,
                        password=UBUNTU_PASS, timeout=15)
            sftp = ssh.open_sftp()
            # 确保远端目录存在
            remote_dir = self.remote.rsplit("/", 1)[0]
            try:
                ssh.exec_command(f"mkdir -p {remote_dir}")
            except Exception:
                pass
            self.log_line.emit(
                f"⬆ 上传: {self.local} → {self.remote}", "info")
            sftp.put(self.local, self.remote)
            sftp.close()
            ssh.close()
            self.log_line.emit(f"━━ 上传完成: {self.remote} ━━", "ok")
            self.finished.emit(0, self.remote)
        except Exception as e:
            self.log_line.emit(f"━━ SFTP 上传失败: {e} ━━", "err")
            self.finished.emit(-1, "")


# ═══════════════════════════════════════════════════════════════
# UI 组件
# ═══════════════════════════════════════════════════════════════
class DecoLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(3)
        self.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {C_BORDER_L}, stop:0.3 {C_GOLD},"
            f"stop:0.7 {C_GOLD}, stop:1 {C_BORDER_L});")


class LogPanel(QTextEdit):
    COLORS = {
        "info": C_TEXT, "cmd": C_ACCENT_L,
        "ok": C_OK, "err": C_ERR, "warn": C_WARN,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 12))
        self.setMinimumHeight(120)
        self.setStyleSheet(f"""
            QTextEdit {{
                background: {C_PANEL_ALT};
                border: 2px solid {C_BORDER};
                border-top: 3px solid {C_GOLD};
                border-radius: 4px; padding: 8px;
                color: {C_TEXT};
                font-family: 'Cascadia Code','Consolas',monospace;
                font-size: 13px;
            }}""")

    def append_log(self, text, level="info"):
        color = self.COLORS.get(level, C_TEXT)
        self.append(f'<span style="color:{color}">{text}</span>')
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_log(self):
        self.clear()


class MetricCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFixedSize(155, 85)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C_PANEL};
                border: 2px solid {C_BORDER};
                border-top: 3px solid {C_GOLD};
                border-radius: 6px;
            }}""")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)
        self.lbl_title = QLabel(title)
        self.lbl_title.setFont(QFont("Microsoft YaHei UI", 10))
        self.lbl_title.setStyleSheet(
            f"color: {C_TEXT_S}; border: none;")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_value = QLabel("—")
        self.lbl_value.setFont(QFont("Consolas", 19, QFont.Bold))
        self.lbl_value.setStyleSheet(
            f"color: {C_ACCENT}; border: none;")
        self.lbl_value.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_value)

    def set_value(self, text, color=None):
        self.lbl_value.setText(text)
        if color:
            self.lbl_value.setStyleSheet(
                f"color: {color}; border: none;")

    def reset(self):
        self.lbl_value.setText("—")
        self.lbl_value.setStyleSheet(
            f"color: {C_ACCENT}; border: none;")


class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.playing = False
        self.total_frames = 0
        self.current_frame = 0
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.display = QLabel()
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setMinimumSize(360, 260)
        self.display.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display.setStyleSheet(f"""
            background: #1a1a2e; border: 2px solid {C_BORDER};
            border-radius: 6px; color: #888; font-size: 14px;
        """)
        self.display.setText("暂无媒体\n执行任务后可在此预览结果")
        lay.addWidget(self.display, stretch=1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.sliderMoved.connect(self._seek)
        lay.addWidget(self.slider)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        self.btn_open = QPushButton("📂 打开")
        self.btn_open.setObjectName("btnGold")
        self.btn_open.clicked.connect(self._open_file)
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setObjectName("btnGold")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setObjectName("btnGold")
        self.btn_stop.clicked.connect(self._stop)
        self.lbl_info = QLabel("0 / 0")
        self.lbl_info.setFont(QFont("Consolas", 12))
        self.lbl_info.setStyleSheet(f"color: {C_TEXT_S};")
        ctrl.addWidget(self.btn_open)
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch()
        ctrl.addWidget(self.lbl_info)
        lay.addLayout(ctrl)

    def load_video(self, path):
        if not _CV2:
            self.display.setText("视频播放需要 opencv-python\n请激活 bishe 环境后重启")
            return
        self._stop()
        if self.cap:
            self.cap.release()
        if not os.path.isfile(path):
            self.display.setText(f"文件不存在:\n{path}")
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.display.setText(f"无法打开:\n{path}")
            return
        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.interval = int(1000 / fps)
        self.slider.setMaximum(max(0, self.total_frames - 1))
        self.current_frame = 0
        self._show_frame(0)
        self.lbl_info.setText(
            f"0 / {self.total_frames}  ({fps:.0f}fps)")

    def load_image(self, path):
        self._stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if not os.path.isfile(path):
            self.display.setText(f"文件不存在:\n{path}")
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.display.setText(f"无法打开:\n{path}")
            return
        scaled = pixmap.scaled(
            self.display.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self.display.setPixmap(scaled)
        self.lbl_info.setText("静态图片")
        self.slider.setMaximum(0)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开媒体文件", str(BOARD_RESULTS_DIR),
            "媒体 (*.mp4 *.avi *.png *.jpg);;所有文件 (*)")
        if path:
            if path.lower().endswith(('.mp4', '.avi', '.mkv')):
                self.load_video(path)
            else:
                self.load_image(path)

    def _toggle_play(self):
        if not self.cap:
            return
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.btn_play.setText("▶ 播放")
        else:
            self.timer.start(self.interval)
            self.playing = True
            self.btn_play.setText("⏸ 暂停")

    def _stop(self):
        self.timer.stop()
        self.playing = False
        self.btn_play.setText("▶ 播放")
        self.current_frame = 0
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._show_frame(0)

    def _seek(self, pos):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            self.current_frame = pos
            self._read_and_show()

    def _next_frame(self):
        if not self.cap:
            return
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self._stop()
            return
        self._read_and_show()

    def _show_frame(self, idx):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._read_and_show()

    def _read_and_show(self):
        ret, frame = self.cap.read()
        if not ret:
            self._stop()
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w,
                       QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.display.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self.display.setPixmap(scaled)
        self.slider.setValue(self.current_frame)
        self.lbl_info.setText(
            f"{self.current_frame} / {self.total_frames}")


class ResultBrowser(QWidget):
    """通用结果文件浏览面板 (仅部署页使用)"""
    def __init__(self, scan_dirs, patterns=("*.png", "*.mp4"),
                 title="◈  结果预览  ◈", parent=None):
        super().__init__(parent)
        self.scan_dirs = scan_dirs
        self.patterns = patterns
        self._build(title)

    def _build(self, title):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 16, 20, 16)

        lbl = QLabel(title)
        lbl.setFont(QFont("Microsoft YaHei UI", 15, QFont.Bold))
        lbl.setStyleSheet(f"color: {C_ACCENT};")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        lay.addWidget(DecoLine())

        self.player = VideoPlayer()
        lay.addWidget(self.player, stretch=1)

        # ── 本次拉取历史 (当前 UI 运行期间累积) ──
        hg = QGroupBox("◈  本次拉取历史  ◈")
        hl = QVBoxLayout()
        self.history_combo = QComboBox()
        self.history_combo.setPlaceholderText("— 本次会话拉取的文件 —")
        hl.addWidget(self.history_combo)
        hb = QHBoxLayout()
        bh = QPushButton("📺  查看")
        bh.setObjectName("btnPrimary")
        bh.clicked.connect(self._view_history)
        hb.addWidget(bh)
        hl.addLayout(hb)
        hg.setLayout(hl)
        lay.addWidget(hg)

        # ── 已有结果文件 (outputs/rv1126b_board_results/ 扫描) ──
        fg = QGroupBox("◈  已有结果文件  ◈")
        fl = QVBoxLayout()
        self.combo = QComboBox()
        fl.addWidget(self.combo)
        rb = QHBoxLayout()
        bv = QPushButton("📺  查看")
        bv.setObjectName("btnPrimary")
        bv.clicked.connect(self._view)
        br = QPushButton("🔄  刷新")
        br.setObjectName("btnGold")
        br.clicked.connect(self.refresh)
        rb.addWidget(bv)
        rb.addWidget(br)
        fl.addLayout(rb)
        fg.setLayout(fl)
        lay.addWidget(fg)
        self.refresh()

    def add_pulled(self, path):
        """拉取完成后调用，追加到本次拉取历史"""
        if not path or not os.path.isfile(path):
            return
        icon = "🎬" if path.lower().endswith(
            ('.mp4', '.avi')) else "🖼"
        label = f"{icon}  {os.path.basename(path)}"
        # 避免重复添加同名路径
        for i in range(self.history_combo.count()):
            if self.history_combo.itemData(i) == path:
                self.history_combo.setCurrentIndex(i)
                return
        self.history_combo.addItem(label, path)
        self.history_combo.setCurrentIndex(
            self.history_combo.count() - 1)

    def _view_history(self):
        p = self.history_combo.currentData()
        if not p:
            QMessageBox.information(
                self, "提示", "本次会话尚未拉取任何文件")
            return
        if not os.path.isfile(p):
            QMessageBox.warning(
                self, "错误", f"文件不存在:\n{p}")
            return
        if p.lower().endswith(('.mp4', '.avi')):
            self.player.load_video(p)
        else:
            self.player.load_image(p)

    def refresh(self):
        self.combo.clear()
        for pat in self.patterns:
            for label, path in find_files(self.scan_dirs, pat):
                icon = "🎬" if path.endswith(('.mp4', '.avi')) \
                    else "🖼"
                self.combo.addItem(f"{icon}  {label}", path)

    def _view(self):
        p = self.combo.currentData()
        if not p:
            QMessageBox.information(
                self, "提示", "没有可查看的文件，先运行任务")
            return
        if not os.path.isfile(p):
            QMessageBox.warning(
                self, "错误", f"文件不存在:\n{p}")
            return
        if p.lower().endswith(('.mp4', '.avi')):
            self.player.load_video(p)
        else:
            self.player.load_image(p)


# ═══════════════════════════════════════════════════════════════
# 数据处理 — 右侧: 数据集概览面板
# ═══════════════════════════════════════════════════════════════
class DatasetOverviewPanel(QWidget):
    """样本图网格 + dataset.yaml 信息 + 类别/标注统计"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 16, 20, 16)

        lbl = QLabel("◈  数据集概览  ◈")
        lbl.setFont(QFont("Microsoft YaHei UI", 15, QFont.Bold))
        lbl.setStyleSheet(f"color: {C_ACCENT};")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        lay.addWidget(DecoLine())

        # ── 样本图网格 (2行×3列) ──
        sg = QGroupBox("◈  样本图预览  ◈")
        self.grid = QGridLayout()
        self.grid.setSpacing(6)
        self.sample_labels = []
        for r in range(2):
            for c in range(3):
                cell = QLabel()
                cell.setAlignment(Qt.AlignCenter)
                cell.setFixedSize(155, 110)
                cell.setStyleSheet(
                    f"background: {C_PANEL_ALT}; "
                    f"border: 1px solid {C_BORDER}; "
                    f"border-radius: 4px; color: #999;")
                cell.setText("—")
                self.grid.addWidget(cell, r, c)
                self.sample_labels.append(cell)
        sg.setLayout(self.grid)
        lay.addWidget(sg)

        # ── 数据集信息 ──
        ig = QGroupBox("◈  数据集信息  ◈")
        il = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Consolas", 12))
        self.info_text.setStyleSheet(
            f"background: {C_PANEL_ALT}; "
            f"border: 1px solid {C_BORDER}; "
            f"border-radius: 4px; padding: 8px;")
        il.addWidget(self.info_text)
        ig.setLayout(il)
        lay.addWidget(ig, stretch=1)

        row = QHBoxLayout()
        btn = QPushButton("🔄  刷新概览")
        btn.setObjectName("btnGold")
        btn.clicked.connect(self.refresh)
        row.addWidget(btn)
        row.addStretch()
        lay.addLayout(row)
        self.refresh()

    def refresh(self):
        yaml_p = ROOT / "data" / "processed" / "flir" / "dataset.yaml"
        if not yaml_p.exists():
            self.info_text.setPlainText(
                "未找到 dataset.yaml\n请先运行 FLIR 数据处理")
            return
        try:
            import yaml
            with open(yaml_p, encoding="utf-8") as f:
                ds = yaml.safe_load(f)
            info = []
            nc = ds.get("nc", "?")
            names = ds.get("names", {})
            info.append(f"类别数:  {nc}")
            if isinstance(names, dict):
                info.append(f"类别名:  {', '.join(names.values())}")
            elif isinstance(names, list):
                info.append(f"类别名:  {', '.join(names)}")
            info.append(f"训练集:  {ds.get('train', '?')}")
            info.append(f"验证集:  {ds.get('val', '?')}")

            train_dir = yaml_p.parent / "images" / "train"
            val_dir = yaml_p.parent / "images" / "val"
            nt = len(list(train_dir.glob("*"))) \
                if train_dir.is_dir() else 0
            nv = len(list(val_dir.glob("*"))) \
                if val_dir.is_dir() else 0
            info.append("")
            info.append(f"训练图像:  {nt} 张")
            info.append(f"验证图像:  {nv} 张")
            info.append(f"总计:      {nt + nv} 张")
            info.append(f"划分比例:  "
                        f"{nt/(nt+nv)*100:.1f}% / "
                        f"{nv/(nt+nv)*100:.1f}%"
                        if nt + nv > 0 else "—")

            # 标注统计 (采样前 300 个标签文件)
            label_dir = yaml_p.parent / "labels" / "train"
            if label_dir.is_dir():
                cls_cnt = {}
                total_boxes = 0
                for lf in sorted(label_dir.glob("*.txt"))[:300]:
                    try:
                        for line in lf.read_text().strip().splitlines():
                            if line.strip():
                                cid = line.split()[0]
                                cls_cnt[cid] = cls_cnt.get(cid, 0) + 1
                                total_boxes += 1
                    except Exception:
                        pass
                info.append(f"\n标注统计 (训练集前 300 文件):")
                info.append(f"  总目标框: {total_boxes}")
                for cid in sorted(cls_cnt):
                    n = cls_cnt[cid]
                    cname = cid
                    if isinstance(names, dict):
                        cname = names.get(int(cid), cid)
                    elif isinstance(names, list):
                        try:
                            cname = names[int(cid)]
                        except (IndexError, ValueError):
                            pass
                    pct = n / total_boxes * 100 if total_boxes else 0
                    info.append(
                        f"  {cname}: {n} ({pct:.1f}%)")
            self.info_text.setPlainText("\n".join(info))
        except Exception as e:
            self.info_text.setPlainText(f"读取失败: {e}")

        # 加载样本图 (训练集随机3 + 验证集随机3)
        imgs = []
        for sub in ["train", "val"]:
            d = yaml_p.parent / "images" / sub
            if d.is_dir():
                all_files = list(d.glob("*"))
                k = min(3, len(all_files))
                if k > 0:
                    for f in random.sample(all_files, k):
                        imgs.append(str(f))
        for i, cell in enumerate(self.sample_labels):
            if i < len(imgs):
                pix = QPixmap(imgs[i])
                if not pix.isNull():
                    cell.setPixmap(pix.scaled(
                        cell.size(), Qt.KeepAspectRatio,
                        Qt.SmoothTransformation))
                else:
                    cell.setText("无法加载")
            else:
                cell.setText("—")


# ═══════════════════════════════════════════════════════════════
# 模型训练 — 右侧: 训练曲线仪表盘
# ═══════════════════════════════════════════════════════════════
class TrainingDashboardPanel(QWidget):
    """实验选择 + 12 种曲线/图片切换 + 末轮 CSV 指标"""
    CURVES = [
        ("📈  训练曲线 (results.png)",   "results.png"),
        ("🎯  混淆矩阵",                "confusion_matrix.png"),
        ("📊  F1 曲线",                 "F1_curve.png"),
        ("📊  PR 曲线",                 "PR_curve.png"),
        ("📊  Precision 曲线",          "P_curve.png"),
        ("📊  Recall 曲线",             "R_curve.png"),
        ("📋  标签分布",                "labels.jpg"),
        ("📋  标签相关性",              "labels_correlogram.jpg"),
        ("🖼  验证预测 batch0",         "val_batch0_pred.jpg"),
        ("🖼  验证预测 batch1",         "val_batch1_pred.jpg"),
        ("🖼  验证真值 batch0",         "val_batch0_labels.jpg"),
        ("🖼  训练样本 batch0",         "train_batch0.jpg"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 16, 20, 16)

        lbl = QLabel("◈  训练曲线仪表盘  ◈")
        lbl.setFont(QFont("Microsoft YaHei UI", 15, QFont.Bold))
        lbl.setStyleSheet(f"color: {C_ACCENT};")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        lay.addWidget(DecoLine())

        # ── 实验 + 曲线选择 ──
        sg = QGroupBox("◈  实验 / 曲线选择  ◈")
        sl = QVBoxLayout()
        self.exp_combo = QComboBox()
        sl.addWidget(self.exp_combo)
        self.curve_combo = QComboBox()
        for label, fname in self.CURVES:
            self.curve_combo.addItem(label, fname)
        sl.addWidget(self.curve_combo)
        row = QHBoxLayout()
        bv = QPushButton("📺  查看")
        bv.setObjectName("btnPrimary")
        bv.clicked.connect(self._view)
        br = QPushButton("🔄  刷新实验")
        br.setObjectName("btnGold")
        br.clicked.connect(self._refresh_exps)
        row.addWidget(bv)
        row.addWidget(br)
        sl.addLayout(row)
        sg.setLayout(sl)
        lay.addWidget(sg)

        # ── 图像显示 ──
        self.display = QLabel()
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display.setMinimumSize(360, 260)
        self.display.setStyleSheet(
            f"background: {C_PANEL_ALT}; "
            f"border: 2px solid {C_BORDER}; "
            f"border-radius: 6px; color: #888; font-size: 14px;")
        self.display.setText("选择实验和曲线类型后点击查看")
        lay.addWidget(self.display, stretch=1)

        # ── 末轮指标 ──
        self.epoch_lbl = QLabel()
        self.epoch_lbl.setFont(QFont("Consolas", 11))
        self.epoch_lbl.setStyleSheet(f"color: {C_TEXT_S};")
        self.epoch_lbl.setWordWrap(True)
        lay.addWidget(self.epoch_lbl)

        self._refresh_exps()

    def _refresh_exps(self):
        self.exp_combo.clear()
        for src_label, src in [
            ("", ABLATION_DIR),
            ("(old) ", ROOT / "outputs" / "ablation_old"),
        ]:
            if src.is_dir():
                for d in sorted(src.iterdir()):
                    if d.is_dir() and (d / "results.png").exists():
                        self.exp_combo.addItem(
                            f"{src_label}{d.name}", str(d))

    def _view(self):
        exp_dir = self.exp_combo.currentData()
        fname = self.curve_combo.currentData()
        if not exp_dir:
            return
        p = Path(exp_dir) / fname
        if not p.exists():
            self.display.setText(f"文件不存在: {fname}")
            return
        pix = QPixmap(str(p))
        if not pix.isNull():
            self.display.setPixmap(pix.scaled(
                self.display.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        # 读 results.csv 末行指标
        csv_p = Path(exp_dir) / "results.csv"
        if csv_p.exists():
            try:
                lines = csv_p.read_text(
                    encoding="utf-8").strip().splitlines()
                if len(lines) >= 2:
                    cols = [c.strip() for c in
                            lines[0].split(",")]
                    vals = [v.strip() for v in
                            lines[-1].split(",")]
                    parts = []
                    for i, c in enumerate(cols):
                        cl = c.lower()
                        if i < len(vals) and any(
                            k in cl for k in [
                                "epoch", "map", "precision",
                                "recall", "loss"]):
                            try:
                                parts.append(
                                    f"{c}={float(vals[i]):.4f}")
                            except ValueError:
                                parts.append(f"{c}={vals[i]}")
                    self.epoch_lbl.setText(
                        "最终 epoch │ " + " │ ".join(parts[:8]))
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# 检测评估 — 右侧: 检测评估看板
# ═══════════════════════════════════════════════════════════════
class DetectionDashboardPanel(QWidget):
    """批次选择 + 对比图表 + summary.csv 表格 + 全局结果图"""
    CHART_TYPES = [
        ("📊  指标柱状图 (barline)",    "journal_barline.png"),
        ("📈  改进分析 (improvement)",   "journal_improvement.png"),
        ("📉  指标变化 (metric_change)", "journal_metric_change.png"),
        ("⚡  效率对比",                 "efficiency_comparison.png"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 16, 20, 16)

        lbl = QLabel("◈  检测评估看板  ◈")
        lbl.setFont(QFont("Microsoft YaHei UI", 15, QFont.Bold))
        lbl.setStyleSheet(f"color: {C_ACCENT};")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        lay.addWidget(DecoLine())

        # ── 批次 + 图表类型 ──
        sg = QGroupBox("◈  评估批次 / 图表  ◈")
        sl = QVBoxLayout()
        self.batch_combo = QComboBox()
        sl.addWidget(self.batch_combo)
        self.chart_combo = QComboBox()
        for label, fname in self.CHART_TYPES:
            self.chart_combo.addItem(label, fname)
        sl.addWidget(self.chart_combo)
        row = QHBoxLayout()
        bv = QPushButton("📺  查看图表")
        bv.setObjectName("btnPrimary")
        bv.clicked.connect(self._view_chart)
        br = QPushButton("🔄  刷新")
        br.setObjectName("btnGold")
        br.clicked.connect(self._refresh)
        row.addWidget(bv)
        row.addWidget(br)
        sl.addLayout(row)
        sg.setLayout(sl)
        lay.addWidget(sg)

        # ── 图表显示 ──
        self.display = QLabel()
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display.setMinimumSize(360, 240)
        self.display.setStyleSheet(
            f"background: {C_PANEL_ALT}; "
            f"border: 2px solid {C_BORDER}; "
            f"border-radius: 6px; color: #888; font-size: 14px;")
        self.display.setText(
            "运行批量检测评估后可查看对比图表")
        lay.addWidget(self.display, stretch=1)

        self._refresh()

    def _refresh(self):
        self.batch_combo.clear()
        if DETECTION_RESULTS_DIR.is_dir():
            for d in sorted(
                    DETECTION_RESULTS_DIR.iterdir(), reverse=True):
                if d.is_dir():
                    self.batch_combo.addItem(d.name, str(d))

    def _view_chart(self):
        bd = self.batch_combo.currentData()
        fname = self.chart_combo.currentData()
        if not bd:
            return
        p = Path(bd) / fname
        if not p.exists():
            self.display.setText(
                f"图表不存在: {fname}\n请先运行 '生成图表'")
            return
        pix = QPixmap(str(p))
        if not pix.isNull():
            self.display.setPixmap(pix.scaled(
                self.display.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))


# ═══════════════════════════════════════════════════════════════
# 跟踪评估 — 右侧: 跟踪分析仪表盘
# ═══════════════════════════════════════════════════════════════
class TrackingDashboardPanel(QWidget):
    """实验→跟踪器→序列 三级选择 + 视频播放 + 逐序列 metrics"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 16, 20, 16)

        lbl = QLabel("◈  跟踪分析仪表盘  ◈")
        lbl.setFont(QFont("Microsoft YaHei UI", 15, QFont.Bold))
        lbl.setStyleSheet(f"color: {C_ACCENT};")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        lay.addWidget(DecoLine())

        # ── 三级选择 ──
        sg = QGroupBox("◈  结果浏览 (实验→跟踪器→序列)  ◈")
        sl = QVBoxLayout()
        sl.setSpacing(8)

        r0 = QHBoxLayout()
        r0.addWidget(QLabel("实验:"))
        self.exp_combo = QComboBox()
        self.exp_combo.currentIndexChanged.connect(self._on_exp_changed)
        r0.addWidget(self.exp_combo, stretch=1)
        sl.addLayout(r0)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("跟踪器:"))
        self.tracker_combo = QComboBox()
        self.tracker_combo.currentIndexChanged.connect(self._on_tracker_changed)
        r1.addWidget(self.tracker_combo, stretch=1)
        sl.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("序列:"))
        self.seq_combo = QComboBox()
        r2.addWidget(self.seq_combo, stretch=1)
        sl.addLayout(r2)

        row = QHBoxLayout()
        bp = QPushButton("▶  播放跟踪视频")
        bp.setObjectName("btnPrimary")
        bp.clicked.connect(self._play_seq)
        br = QPushButton("🔄  刷新")
        br.setObjectName("btnGold")
        br.clicked.connect(self._refresh)
        row.addWidget(bp)
        row.addWidget(br)
        sl.addLayout(row)
        sg.setLayout(sl)
        lay.addWidget(sg)

        # ── 视频播放 ──
        self.player = VideoPlayer()
        lay.addWidget(self.player, stretch=1)

        self._refresh()

    def _refresh(self):
        self.exp_combo.clear()
        base = TRACKING_RESULTS_DIR
        if base.is_dir():
            for batch in sorted(base.iterdir()):
                if not batch.is_dir():
                    continue
                for exp in sorted(batch.iterdir()):
                    if exp.is_dir() and \
                            exp.name.startswith("ablation_"):
                        self.exp_combo.addItem(
                            f"{batch.name} / {exp.name}",
                            str(exp))

    def _on_exp_changed(self):
        self.tracker_combo.clear()
        exp_dir = self.exp_combo.currentData()
        if not exp_dir:
            return
        p = Path(exp_dir)
        if p.is_dir():
            for d in sorted(p.iterdir()):
                if d.is_dir() and any(
                        t in d.name for t in TRACKERS):
                    self.tracker_combo.addItem(
                        d.name, str(d))

    def _on_tracker_changed(self):
        self.seq_combo.clear()
        td = self.tracker_combo.currentData()
        if not td:
            return
        p = Path(td)
        if p.is_dir():
            for d in sorted(p.iterdir()):
                if d.is_dir():
                    vid = d / "result.mp4"
                    mj = d / "metrics.json"
                    if vid.exists() or mj.exists():
                        self.seq_combo.addItem(
                            d.name, str(d))

    def _play_seq(self):
        seq_dir = self.seq_combo.currentData()
        if not seq_dir:
            return
        sd = Path(seq_dir)
        vid = sd / "result.mp4"
        if vid.exists():
            self.player.load_video(str(vid))


# ═══════════════════════════════════════════════════════════════
# 标签页基类 (统一 Splitter 布局)
# ═══════════════════════════════════════════════════════════════
class BaseSplitTab(QWidget):
    """左面板 (参数+操作+日志) + 右面板 (结果预览)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)

        self._left = QWidget()
        self._left_layout = QVBoxLayout(self._left)
        self._left_layout.setContentsMargins(20, 16, 8, 16)
        self._left_layout.setSpacing(10)
        self._build_left(self._left_layout)

        reset_row = QHBoxLayout()
        reset_row.addStretch()
        self._btn_reset = QPushButton("🔄  复位")
        self._btn_reset.setObjectName("btnGold")
        self._btn_reset.clicked.connect(self._reset)
        reset_row.addWidget(self._btn_reset)
        self._left_layout.addLayout(reset_row)

        self.log = LogPanel()
        self._left_layout.addWidget(self.log, stretch=1)

        self._right = self._build_right()

        splitter.addWidget(self._left)
        splitter.addWidget(self._right)
        splitter.setSizes([520, 480])

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

    def _build_left(self, layout):
        """子类覆盖: 向 layout 添加参数面板"""
        pass

    def _build_right(self):
        """子类覆盖: 返回右侧预览 Widget"""
        return QWidget()

    def _exec(self, cmd, btn):
        if self.worker and self.worker.isRunning():
            self.log.append_log("⚠ 上一个任务仍在运行", "warn")
            return
        self.log.clear_log()
        btn.setEnabled(False)
        self.worker = LocalCmdWorker(cmd)
        self.worker.log_line.connect(self.log.append_log)
        self.worker.finished.connect(
            lambda _: btn.setEnabled(True))
        self.worker.start()

    def _reset(self):
        self.log.clear_log()
        self._on_reset()

    def _on_reset(self):
        pass


# ═══════════════════════════════════════════════════════════════
# 标签页 1: 数据处理
# ═══════════════════════════════════════════════════════════════
class DataTab(BaseSplitTab):
    def _build_left(self, lay):
        # ── FLIR ──
        g1 = QGroupBox("◈  FLIR 数据集预处理  ◈")
        gl = QGridLayout()
        gl.setSpacing(10)
        gl.addWidget(QLabel("原始数据目录:"), 0, 0)
        self.flir_input = QLineEdit("data/raw/flir")
        gl.addWidget(self.flir_input, 0, 1)
        b = QPushButton("浏览")
        b.setObjectName("btnGold")
        b.setFixedWidth(70)
        b.clicked.connect(lambda: self._browse(self.flir_input))
        gl.addWidget(b, 0, 2)
        self.btn_flir = QPushButton("▶  处理 FLIR 数据集")
        self.btn_flir.setObjectName("btnPrimary")
        self.btn_flir.clicked.connect(self._run_flir)
        gl.addWidget(self.btn_flir, 1, 0, 1, 3)
        g1.setLayout(gl)
        lay.addWidget(g1)
        lay.addWidget(DecoLine())

        # ── 图像→视频 ──
        g2 = QGroupBox("◈  图像序列转视频  ◈")
        gl2 = QGridLayout()
        gl2.setSpacing(10)
        gl2.addWidget(QLabel("图像目录:"), 0, 0)
        self.img_dir = QLineEdit()
        gl2.addWidget(self.img_dir, 0, 1)
        b2 = QPushButton("浏览")
        b2.setObjectName("btnGold")
        b2.setFixedWidth(70)
        b2.clicked.connect(lambda: self._browse(self.img_dir))
        gl2.addWidget(b2, 0, 2)
        gl2.addWidget(QLabel("输出目录:"), 1, 0)
        self.vid_out = QLineEdit("data/videos/thermal_test")
        gl2.addWidget(self.vid_out, 1, 1, 1, 2)
        gl2.addWidget(QLabel("FPS:"), 2, 0)
        self.fps = QSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setValue(30)
        gl2.addWidget(self.fps, 2, 1, 1, 2)
        self.btn_vid = QPushButton("▶  转换为视频")
        self.btn_vid.setObjectName("btnPrimary")
        self.btn_vid.clicked.connect(self._run_vid)
        gl2.addWidget(self.btn_vid, 3, 0, 1, 3)
        g2.setLayout(gl2)
        lay.addWidget(g2)
        lay.addWidget(DecoLine())

        # ── 信息卡 ──
        mc = QHBoxLayout()
        mc.setSpacing(12)
        self.mc_imgs = MetricCard("图像数")
        self.mc_train = MetricCard("训练集")
        self.mc_val = MetricCard("验证集")
        self.mc_cls = MetricCard("类别数")
        mc.addWidget(self.mc_imgs)
        mc.addWidget(self.mc_train)
        mc.addWidget(self.mc_val)
        mc.addWidget(self.mc_cls)
        mc.addStretch()
        lay.addLayout(mc)

        # ── 扫描按钮 ──
        self.btn_scan = QPushButton("🔍  扫描已处理数据集")
        self.btn_scan.setObjectName("btnGold")
        self.btn_scan.clicked.connect(self._scan_dataset)
        lay.addWidget(self.btn_scan)

    def _build_right(self):
        return DatasetOverviewPanel()

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(
            self, "选择目录", str(ROOT))
        if d:
            le.setText(d)

    def _run_flir(self):
        p = self.flir_input.text().strip()
        if not p:
            QMessageBox.warning(self, "提示", "请填写原始数据目录")
            return
        if not path_valid(p, "dir"):
            QMessageBox.warning(
                self, "错误", f"目录不存在: {p}")
            return
        self._exec(
            f'python scripts/data/prepare_flir.py '
            f'--input "{p}"', self.btn_flir)

    def _run_vid(self):
        p = self.img_dir.text().strip()
        if not p:
            QMessageBox.warning(self, "提示", "请填写图像目录")
            return
        if not path_valid(p, "dir"):
            QMessageBox.warning(
                self, "错误", f"目录不存在: {p}")
            return
        self._exec(
            f'python scripts/data/images_to_video.py '
            f'--input "{p}" '
            f'--output "{self.vid_out.text()}" '
            f'--fps {self.fps.value()}', self.btn_vid)

    def _scan_dataset(self):
        yaml_p = ROOT / "data" / "processed" / "flir" / "dataset.yaml"
        if not yaml_p.exists():
            self.log.append_log(
                "未找到 dataset.yaml，请先处理 FLIR 数据集", "warn")
            return
        try:
            import yaml
            with open(yaml_p, encoding="utf-8") as f:
                ds = yaml.safe_load(f)
            nc = ds.get("nc", "?")
            self.mc_cls.set_value(str(nc), C_ACCENT)
            train_dir = yaml_p.parent / "images" / "train"
            val_dir = yaml_p.parent / "images" / "val"
            nt = len(list(train_dir.glob("*"))) if train_dir.is_dir() else 0
            nv = len(list(val_dir.glob("*"))) if val_dir.is_dir() else 0
            self.mc_train.set_value(str(nt), C_ACCENT)
            self.mc_val.set_value(str(nv), C_ACCENT)
            self.mc_imgs.set_value(str(nt + nv), C_ACCENT)
            self.log.append_log(
                f"数据集: {nt} 训练 + {nv} 验证, "
                f"{nc} 类别", "ok")
        except Exception as e:
            self.log.append_log(f"扫描失败: {e}", "err")

    def _on_reset(self):
        self.flir_input.setText("data/raw/flir")
        self.img_dir.clear()
        self.vid_out.setText("data/videos/thermal_test")
        self.fps.setValue(30)
        for mc in [self.mc_imgs, self.mc_train,
                   self.mc_val, self.mc_cls]:
            mc.reset()


# ═══════════════════════════════════════════════════════════════
# 标签页 2: 模型训练
# ═══════════════════════════════════════════════════════════════
class TrainTab(BaseSplitTab):
    def _build_left(self, lay):
        # ── 单模型 ──
        g1 = QGroupBox("◈  单模型训练  ◈")
        gl = QGridLayout()
        gl.setSpacing(10)
        gl.addWidget(QLabel("配置文件:"), 0, 0)
        self.cfg = QLineEdit("configs/train_config.yaml")
        gl.addWidget(self.cfg, 0, 1)
        bc = QPushButton("浏览")
        bc.setObjectName("btnGold")
        bc.setFixedWidth(70)
        bc.clicked.connect(self._browse_cfg)
        gl.addWidget(bc, 0, 2)
        self._spins = []
        for i, (lbl, lo, hi, val) in enumerate([
            ("Epochs:", 1, 1000, 100),
            ("Batch:", 1, 128, 16),
            ("Img Size:", 320, 1280, 640),
        ]):
            gl.addWidget(QLabel(lbl), i + 1, 0)
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(val)
            if lbl == "Img Size:":
                spin.setSingleStep(32)
            gl.addWidget(spin, i + 1, 1, 1, 2)
            self._spins.append(spin)
        self.btn_train = QPushButton("▶  开始训练")
        self.btn_train.setObjectName("btnPrimary")
        self.btn_train.clicked.connect(self._run_train)
        gl.addWidget(self.btn_train, 4, 0, 1, 3)
        g1.setLayout(gl)
        lay.addWidget(g1)
        lay.addWidget(DecoLine())

        # ── 消融 ──
        g2 = QGroupBox("◈  消融实验  ◈")
        gl2 = QGridLayout()
        gl2.setSpacing(10)
        gl2.addWidget(QLabel("Profile:"), 0, 0)
        self.profile = QComboBox()
        self.profile.addItem(
            "controlled — 严格控变量", "controlled")
        self.profile.addItem(
            "optimal — 独立最优参数", "optimal")
        gl2.addWidget(self.profile, 0, 1, 1, 2)
        gl2.addWidget(QLabel("仅运行:"), 1, 0)
        self.only = QLineEdit()
        self.only.setPlaceholderText(
            "留空=全部 | exp7=仅 EIoU")
        gl2.addWidget(self.only, 1, 1, 1, 2)
        self.btn_abl = QPushButton("▶  开始消融训练")
        self.btn_abl.setObjectName("btnPrimary")
        self.btn_abl.clicked.connect(self._run_abl)
        gl2.addWidget(self.btn_abl, 2, 0, 1, 3)
        g2.setLayout(gl2)
        lay.addWidget(g2)
        lay.addWidget(DecoLine())

        # ── 指标 ──
        mc = QHBoxLayout()
        mc.setSpacing(12)
        self.mc_exp = MetricCard("实验数")
        self.mc_best = MetricCard("最佳 mAP50")
        self.mc_name = MetricCard("最佳模型")
        mc.addWidget(self.mc_exp)
        mc.addWidget(self.mc_best)
        mc.addWidget(self.mc_name)
        mc.addStretch()
        lay.addLayout(mc)

        # ── 扫描 ──
        self.btn_scan = QPushButton(
            "🔍  扫描消融实验结果")
        self.btn_scan.setObjectName("btnGold")
        self.btn_scan.clicked.connect(self._scan_ablation)
        lay.addWidget(self.btn_scan)

    def _build_right(self):
        return TrainingDashboardPanel()

    def _browse_cfg(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择配置", str(ROOT / "configs"), "*.yaml")
        if p:
            self.cfg.setText(p)

    def _run_train(self):
        cfg = self.cfg.text().strip()
        if not cfg:
            QMessageBox.warning(self, "提示", "请填写配置文件路径")
            return
        if not path_valid(cfg):
            QMessageBox.warning(
                self, "错误", f"配置文件不存在: {cfg}")
            return
        self._exec(
            f'python scripts/train/train_yolov5.py '
            f'--config "{cfg}" '
            f'--epochs {self._spins[0].value()} '
            f'--batch-size {self._spins[1].value()} '
            f'--img-size {self._spins[2].value()}',
            self.btn_train)

    def _run_abl(self):
        cmd = (f'python scripts/train/train_ablation.py '
               f'--profile {self.profile.currentData()}')
        o = self.only.text().strip()
        if o:
            cmd += f' --only {o}'
        self._exec(cmd, self.btn_abl)

    def _scan_ablation(self):
        if not ABLATION_DIR.is_dir():
            self.log.append_log(
                "消融目录不存在，请先运行训练", "warn")
            return
        exps = sorted([
            d for d in ABLATION_DIR.iterdir()
            if d.is_dir() and (d / "weights" / "best.pt").exists()
        ])
        self.mc_exp.set_value(str(len(exps)), C_ACCENT)
        best_map, best_name = 0.0, "—"
        for exp in exps:
            csv = exp / "results.csv"
            if csv.exists():
                try:
                    lines = csv.read_text(
                        encoding="utf-8").strip().splitlines()
                    if len(lines) > 1:
                        last = lines[-1].split(",")
                        # mAP@0.5 is typically column index 6
                        for idx in [6, 7, 8]:
                            try:
                                v = float(last[idx].strip())
                                if 0 < v <= 1 and v > best_map:
                                    best_map = v
                                    best_name = exp.name
                            except (IndexError, ValueError):
                                pass
                except Exception:
                    pass
        if best_map > 0:
            self.mc_best.set_value(
                f"{best_map:.3f}", C_OK)
            short = best_name.replace(
                "ablation_", "").replace("exp", "E")
            self.mc_name.set_value(short[:8], C_ACCENT)
        self.log.append_log(
            f"共 {len(exps)} 个完成的实验", "ok")

    def _on_reset(self):
        self.cfg.setText("configs/train_config.yaml")
        self._spins[0].setValue(100)
        self._spins[1].setValue(16)
        self._spins[2].setValue(640)
        self.profile.setCurrentIndex(0)
        self.only.clear()
        for mc in [self.mc_exp, self.mc_best, self.mc_name]:
            mc.reset()


# ═══════════════════════════════════════════════════════════════
# 标签页 3: 检测评估
# ═══════════════════════════════════════════════════════════════
class DetectionTab(BaseSplitTab):
    def _build_left(self, lay):
        g = QGroupBox("◈  检测评估参数  ◈")
        pg = QGridLayout()
        pg.setSpacing(10)
        pg.addWidget(QLabel("模式:"), 0, 0)
        self.mode = QComboBox()
        self.mode.addItem("metric — P/R/mAP", "metric")
        self.mode.addItem("speed — 推理速度", "speed")
        pg.addWidget(self.mode, 0, 1, 1, 2)

        pg.addWidget(QLabel("权重:"), 1, 0)
        self.weights = QComboBox()
        self.weights.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed)
        for name, path in find_weights():
            self.weights.addItem(name, path)
        pg.addWidget(self.weights, 1, 1)
        bw = QPushButton("浏览")
        bw.setObjectName("btnGold")
        bw.setFixedWidth(70)
        bw.clicked.connect(self._browse_w)
        pg.addWidget(bw, 1, 2)

        self._params = []
        for i, (lbl, lo, hi, val, dec) in enumerate([
            ("Conf:", 0.001, 1.0, 0.001, 3),
            ("IoU:",  0.1,   1.0, 0.6,   2),
            ("Img:",  320,   1280, 640,  0),
            ("Batch:", 1,    128,  32,   0),
        ]):
            pg.addWidget(QLabel(lbl), 2 + i, 0)
            if dec > 0:
                s = QDoubleSpinBox()
                s.setRange(lo, hi)
                s.setValue(val)
                s.setDecimals(dec)
                s.setSingleStep(0.01 if dec >= 3 else 0.05)
            else:
                s = QSpinBox()
                s.setRange(int(lo), int(hi))
                s.setValue(int(val))
                if lbl == "Img:":
                    s.setSingleStep(32)
            pg.addWidget(s, 2 + i, 1, 1, 2)
            self._params.append(s)

        self.chk_batch = QCheckBox("批量评估全部消融实验")
        pg.addWidget(self.chk_batch, 6, 0, 1, 3)
        g.setLayout(pg)
        lay.addWidget(g)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.btn_eval = QPushButton("▶  检测评估")
        self.btn_eval.setObjectName("btnPrimary")
        self.btn_eval.clicked.connect(self._run_eval)
        self.btn_plot = QPushButton("📊  生成图表")
        self.btn_plot.setObjectName("btnGold")
        self.btn_plot.clicked.connect(self._run_plot)
        row.addWidget(self.btn_eval)
        row.addWidget(self.btn_plot)
        row.addStretch()
        lay.addLayout(row)
        lay.addWidget(DecoLine())

        # ── 指标卡 ──
        mc = QHBoxLayout()
        mc.setSpacing(12)
        self.mc_map50 = MetricCard("mAP@50")
        self.mc_map5095 = MetricCard("mAP@50:95")
        self.mc_prec = MetricCard("Precision")
        self.mc_recall = MetricCard("Recall")
        mc.addWidget(self.mc_map50)
        mc.addWidget(self.mc_map5095)
        mc.addWidget(self.mc_prec)
        mc.addWidget(self.mc_recall)
        mc.addStretch()
        lay.addLayout(mc)

    def _build_right(self):
        self._det_dashboard = DetectionDashboardPanel()
        return self._det_dashboard

    def _browse_w(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择权重", str(ROOT / "outputs"), "*.pt")
        if p:
            self.weights.addItem(Path(p).stem, p)
            self.weights.setCurrentIndex(
                self.weights.count() - 1)

    def _run_eval(self):
        w = self.weights.currentData()
        if not w and not self.chk_batch.isChecked():
            QMessageBox.warning(self, "提示", "请选择权重")
            return
        if w and not self.chk_batch.isChecked():
            if not os.path.isfile(w):
                QMessageBox.warning(
                    self, "错误", f"权重文件不存在:\n{w}")
                return
        cmd = (f'python scripts/evaluate/eval_detection.py '
               f'--config configs/eval_detection.yaml '
               f'--mode {self.mode.currentData()} '
               f'--conf-thres {self._params[0].value()} '
               f'--iou-thres {self._params[1].value()} '
               f'--img-size {self._params[2].value()} '
               f'--batch-size {self._params[3].value()}')
        if w and not self.chk_batch.isChecked():
            cmd += f' --weights "{w}"'
        if self.chk_batch.isChecked():
            cmd += ' --batch-eval'
        # 监听输出解析指标
        self.log.clear_log()
        self.btn_eval.setEnabled(False)
        self.worker = LocalCmdWorker(cmd)
        self.worker.log_line.connect(self.log.append_log)
        self.worker.log_line.connect(self._parse_det)
        def _after(code):
            self.btn_eval.setEnabled(True)
            if hasattr(self, '_det_dashboard'):
                self._det_dashboard._refresh()
        self.worker.finished.connect(_after)
        self.worker.start()

    def _parse_det(self, text, level):
        # YOLOv5 val 输出格式: all  images  labels  P  R  mAP50  mAP50-95
        m = re.search(
            r'all\s+\d+\s+\d+\s+'
            r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
            text)
        if m:
            p, r, m50, m5095 = [float(x) for x in m.groups()]
            self.mc_prec.set_value(f"{p:.3f}", C_ACCENT)
            self.mc_recall.set_value(f"{r:.3f}", C_ACCENT)
            self.mc_map50.set_value(f"{m50:.3f}", C_OK)
            self.mc_map5095.set_value(f"{m5095:.3f}", C_ACCENT)

    def _run_plot(self):
        if not DETECTION_RESULTS_DIR.is_dir():
            QMessageBox.warning(
                self, "提示",
                "请先运行检测评估生成结果数据")
            return
        dirs = sorted(
            [d for d in DETECTION_RESULTS_DIR.iterdir()
             if d.is_dir()], reverse=True)
        if not dirs:
            QMessageBox.warning(
                self, "提示",
                "outputs/detection 下无批次目录\n请先运行检测评估")
            return
        latest = dirs[0]
        if not (latest / "summary.csv").exists():
            QMessageBox.warning(
                self, "提示",
                f"最新批次 {latest.name} 中无 summary.csv\n"
                f"请先运行批量检测评估（勾选 '批量评估全部消融实验'）")
            return
        self._exec(
            f'python scripts/evaluate/plot_eval_summary.py '
            f'--config configs/plot_eval_summary.yaml '
            f'--input-dir "{latest}"',
            self.btn_plot)

    def _on_reset(self):
        self.mode.setCurrentIndex(0)
        self._params[0].setValue(0.001)
        self._params[1].setValue(0.6)
        self._params[2].setValue(640)
        self._params[3].setValue(32)
        self.chk_batch.setChecked(False)
        for mc in [self.mc_map50, self.mc_map5095,
                   self.mc_prec, self.mc_recall]:
            mc.reset()


# ═══════════════════════════════════════════════════════════════
# 标签页 4: 跟踪评估
# ═══════════════════════════════════════════════════════════════
class TrackingTab(BaseSplitTab):
    def _build_left(self, lay):
        g = QGroupBox("◈  跟踪评估参数  ◈")
        pg = QGridLayout()
        pg.setSpacing(10)

        pg.addWidget(QLabel("权重:"), 0, 0)
        self.weights = QComboBox()
        self.weights.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed)
        for name, path in find_weights():
            self.weights.addItem(name, path)
        pg.addWidget(self.weights, 0, 1)
        bw = QPushButton("浏览")
        bw.setObjectName("btnGold")
        bw.setFixedWidth(70)
        bw.clicked.connect(self._browse_w)
        pg.addWidget(bw, 0, 2)

        pg.addWidget(QLabel("跟踪器:"), 1, 0)
        self.tracker = QComboBox()
        for t in TRACKERS:
            self.tracker.addItem(t)
        pg.addWidget(self.tracker, 1, 1, 1, 2)

        pg.addWidget(QLabel("数据源:"), 2, 0)
        self.data = QLineEdit("data/videos/thermal_test")
        pg.addWidget(self.data, 2, 1)
        bd = QPushButton("浏览")
        bd.setObjectName("btnGold")
        bd.setFixedWidth(70)
        bd.clicked.connect(
            lambda: self._browse_dir(self.data))
        pg.addWidget(bd, 2, 2)

        self._tparams = []
        for i, (lbl, lo, hi, val) in enumerate([
            ("Conf:", 0.01, 1.0, 0.25),
            ("NMS:",  0.1,  1.0, 0.45),
        ]):
            pg.addWidget(QLabel(lbl), 3 + i, 0)
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(0.05)
            pg.addWidget(s, 3 + i, 1, 1, 2)
            self._tparams.append(s)
        pg.addWidget(QLabel("Img:"), 5, 0)
        self.img_size = QSpinBox()
        self.img_size.setRange(320, 1280)
        self.img_size.setValue(640)
        self.img_size.setSingleStep(32)
        pg.addWidget(self.img_size, 5, 1, 1, 2)

        pg.addWidget(QLabel("输出:"), 6, 0)
        self.output = QLineEdit("outputs/tracking/current")
        pg.addWidget(self.output, 6, 1, 1, 2)

        chk = QHBoxLayout()
        self.chk_half = QCheckBox("FP16")
        self.chk_half.setChecked(True)
        self.chk_vid = QCheckBox("保存视频")
        self.chk_txt = QCheckBox("保存 MOT txt")
        self.chk_overlay = QCheckBox("绘制框")
        self.chk_overlay.setChecked(True)
        for c in [self.chk_half, self.chk_vid,
                   self.chk_txt, self.chk_overlay]:
            chk.addWidget(c)
        chk.addStretch()
        pg.addLayout(chk, 7, 0, 1, 3)
        g.setLayout(pg)
        lay.addWidget(g)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.btn_run = QPushButton("▶  运行跟踪")
        self.btn_run.setObjectName("btnPrimary")
        self.btn_run.clicked.connect(self._run)
        row.addWidget(self.btn_run)
        row.addStretch()
        lay.addLayout(row)
        lay.addWidget(DecoLine())

        # ── 指标卡 ──
        mc = QHBoxLayout()
        mc.setSpacing(12)
        self.mc_mota = MetricCard("MOTA")
        self.mc_idf1 = MetricCard("IDF1")
        self.mc_fps = MetricCard("Track FPS")
        self.mc_idsw = MetricCard("ID Switch")
        mc.addWidget(self.mc_mota)
        mc.addWidget(self.mc_idf1)
        mc.addWidget(self.mc_fps)
        mc.addWidget(self.mc_idsw)
        mc.addStretch()
        lay.addLayout(mc)

    def _build_right(self):
        self._trk_dashboard = TrackingDashboardPanel()
        return self._trk_dashboard

    def _browse_w(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择权重", str(ROOT / "outputs"), "*.pt")
        if p:
            self.weights.addItem(Path(p).stem, p)
            self.weights.setCurrentIndex(
                self.weights.count() - 1)

    def _browse_dir(self, le):
        d = QFileDialog.getExistingDirectory(
            self, "选择目录", str(ROOT))
        if d:
            le.setText(d)

    def _run(self):
        w = self.weights.currentData()
        if not w:
            QMessageBox.warning(self, "提示", "请选择权重")
            return
        if not os.path.isfile(w):
            QMessageBox.warning(
                self, "错误", f"权重不存在:\n{w}")
            return
        data = self.data.text().strip()
        if not data or not path_valid(data, "dir"):
            QMessageBox.warning(
                self, "错误", f"数据源目录不存在: {data}")
            return
        cmd = (f'python scripts/evaluate/eval_tracking.py '
               f'--config configs/tracking_config.yaml '
               f'--weights "{w}" '
               f'--tracker {self.tracker.currentText()} '
               f'--data "{data}" '
               f'--conf-thres {self._tparams[0].value()} '
               f'--nms-thres {self._tparams[1].value()} '
               f'--img-size {self.img_size.value()} '
               f'--output "{self.output.text()}"')
        if self.chk_half.isChecked():
            cmd += ' --half'
        if not self.chk_vid.isChecked():
            cmd += ' --no-save-vid'
        if not self.chk_txt.isChecked():
            cmd += ' --no-save-txt'
        if not self.chk_overlay.isChecked():
            cmd += ' --no-overlay'
        self.log.clear_log()
        self.btn_run.setEnabled(False)
        self.worker = LocalCmdWorker(cmd)
        self.worker.log_line.connect(self.log.append_log)
        self.worker.log_line.connect(self._parse_track)
        def _after(code):
            self.btn_run.setEnabled(True)
            if hasattr(self, '_trk_dashboard'):
                self._trk_dashboard._refresh()
        self.worker.finished.connect(_after)
        self.worker.start()

    def _parse_track(self, text, level):
        m = re.search(r'MOTA[:\s]+([\d.]+)', text, re.I)
        if m:
            self.mc_mota.set_value(
                f"{float(m.group(1)):.1f}", C_OK)
        m = re.search(r'IDF1[:\s]+([\d.]+)', text, re.I)
        if m:
            self.mc_idf1.set_value(
                f"{float(m.group(1)):.1f}", C_ACCENT)
        m = re.search(r'FPS[:\s]+([\d.]+)', text, re.I)
        if m:
            self.mc_fps.set_value(
                f"{float(m.group(1)):.1f}", C_ACCENT)
        m = re.search(r'ID.?Sw\w*[:\s]+(\d+)', text, re.I)
        if m:
            self.mc_idsw.set_value(m.group(1), C_WARN)

    def _on_reset(self):
        self.data.setText("data/videos/thermal_test")
        self._tparams[0].setValue(0.25)
        self._tparams[1].setValue(0.45)
        self.img_size.setValue(640)
        self.output.setText("outputs/tracking/current")
        self.chk_half.setChecked(True)
        self.chk_vid.setChecked(False)
        self.chk_txt.setChecked(False)
        self.chk_overlay.setChecked(True)
        for mc in [self.mc_mota, self.mc_idf1,
                   self.mc_fps, self.mc_idsw]:
            mc.reset()


# ═══════════════════════════════════════════════════════════════
# 标签页 5: 板端部署
# ═══════════════════════════════════════════════════════════════
class DeployTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.sftp_worker = None
        self._session_metrics = {}   # 当前一次推理会话的指标缓存
        self._infer_type = ""        # "video" / "image"
        self._build()

    def _build(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)

        # ════════ 左面板 ════════
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(20, 16, 8, 16)
        ll.setSpacing(10)

        # 连接状态
        sg = QGroupBox("◈  板端连接  ◈")
        sr = QHBoxLayout()
        self.lbl_board = QLabel("Board  ●  —")
        self.lbl_board.setFont(
            QFont("Microsoft YaHei UI", 13, QFont.Bold))
        self.lbl_npu = QLabel("NPU  —")
        self.lbl_npu.setFont(QFont("Microsoft YaHei UI", 13))
        btn_status = QPushButton("🔄 检测连接")
        btn_status.setObjectName("btnGold")
        btn_status.clicked.connect(self._check_status)
        sr.addWidget(self.lbl_board)
        sr.addWidget(self.lbl_npu)
        sr.addStretch()
        sr.addWidget(btn_status)
        sg.setLayout(sr)
        ll.addWidget(sg)

        # 参数
        pg = QGroupBox("◈  板端检测参数  ◈")
        gl = QGridLayout()
        gl.setSpacing(10)
        gl.addWidget(QLabel("RKNN 模型:"), 0, 0)
        self.model = QComboBox()
        for name, key in BOARD_MODELS:
            self.model.addItem(f"{name}  ({key})", key)
        gl.addWidget(self.model, 0, 1, 1, 2)
        gl.addWidget(QLabel("测试视频:"), 1, 0)
        self.video = QComboBox()
        for label, val in BOARD_VIDEOS:
            self.video.addItem(label, val)
        gl.addWidget(self.video, 1, 1, 1, 2)
        gl.addWidget(QLabel("测试图片:"), 2, 0)
        self.image = QComboBox()
        self.image.addItem("全部 testdata/", "ALL")
        for label, val in BOARD_IMAGES:
            self.image.addItem(label, val)
        gl.addWidget(self.image, 2, 1, 1, 2)
        gl.addWidget(QLabel("Conf:"), 3, 0)
        self.conf = QLineEdit()
        self.conf.setPlaceholderText("auto (0.25)")
        gl.addWidget(self.conf, 3, 1, 1, 2)
        gl.addWidget(QLabel("NMS:"), 4, 0)
        self.nms = QLineEdit()
        self.nms.setPlaceholderText("auto (0.45)")
        gl.addWidget(self.nms, 4, 1, 1, 2)
        pg.setLayout(gl)
        ll.addWidget(pg)

        # 操作
        og = QGroupBox("◈  操作  ◈")
        ol = QVBoxLayout()
        r1 = QHBoxLayout()
        self.btn_img = QPushButton("▶  图片检测")
        self.btn_img.setObjectName("btnPrimary")
        self.btn_img.clicked.connect(self._run_image)
        self.btn_vid = QPushButton("▶  视频检测")
        self.btn_vid.setObjectName("btnPrimary")
        self.btn_vid.clicked.connect(self._run_video)
        r1.addWidget(self.btn_img)
        r1.addWidget(self.btn_vid)
        ol.addLayout(r1)

        r2 = QHBoxLayout()
        self.btn_pull_vid = QPushButton("⬇  拉取结果视频")
        self.btn_pull_vid.setObjectName("btnGold")
        self.btn_pull_vid.clicked.connect(self._pull_video)
        self.btn_pull_img = QPushButton("⬇  拉取结果图片")
        self.btn_pull_img.setObjectName("btnGold")
        self.btn_pull_img.clicked.connect(self._pull_images)
        r2.addWidget(self.btn_pull_vid)
        r2.addWidget(self.btn_pull_img)
        ol.addLayout(r2)

        og.setLayout(ol)
        ll.addWidget(og)

        # 指标卡
        ll.addWidget(DecoLine())
        mc = QHBoxLayout()
        mc.setSpacing(12)
        self.mc_npu = MetricCard("NPU 延迟")
        self.mc_fps = MetricCard("NPU FPS")
        self.mc_e2e = MetricCard("平均推理ms")
        self.mc_det = MetricCard("检测数")
        self.mc_track = MetricCard("轨迹展示")
        mc.addWidget(self.mc_npu)
        mc.addWidget(self.mc_fps)
        mc.addWidget(self.mc_e2e)
        mc.addWidget(self.mc_det)
        mc.addWidget(self.mc_track)
        mc.addStretch()
        ll.addLayout(mc)

        reset_row = QHBoxLayout()
        btn_reset = QPushButton("🔄  复位")
        btn_reset.setObjectName("btnGold")
        btn_reset.clicked.connect(self._reset_deploy)
        self.btn_view_log = QPushButton("📋  查看推理日志")
        self.btn_view_log.setObjectName("btnGold")
        self.btn_view_log.clicked.connect(self._open_inference_log)
        reset_row.addWidget(btn_reset)
        reset_row.addWidget(self.btn_view_log)
        reset_row.addStretch()
        ll.addLayout(reset_row)

        self.log = LogPanel()
        ll.addWidget(self.log, stretch=1)

        # ════════ 右面板 ════════
        self.browser = ResultBrowser(
            [BOARD_RESULTS_DIR],
            ("*.mp4", "*.png", "*.jpg"),
            "◈  板端结果预览  ◈")

        splitter.addWidget(left)
        splitter.addWidget(self.browser)
        splitter.setSizes([520, 480])

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

    # ── 公共 ──
    def _adb_cmd(self, board_cmd):
        return (f'adb -s {ADB_SERIAL} shell '
                f'"cd {BOARD_DEPLOY} && '
                f'export LD_LIBRARY_PATH=./lib && '
                f'{board_cmd}"')

    def _all_btns(self):
        return [self.btn_img, self.btn_vid,
                self.btn_pull_vid, self.btn_pull_img]

    def _btns_enabled(self, v):
        for b in self._all_btns():
            b.setEnabled(v)

    def _ssh_run(self, cmd, after_cb=None):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "忙", "上一个任务仍在运行")
            return
        self.log.clear_log()
        self._btns_enabled(False)
        self.worker = SSHWorker(cmd)
        self.worker.log_line.connect(self._on_log)
        if after_cb:
            self.worker.finished.connect(after_cb)
        else:
            self.worker.finished.connect(
                lambda _: self._btns_enabled(True))
        self.worker.start()

    def _on_log(self, text, level):
        # 过滤板端 C++ 程序的 fill dst image 咨骚日志
        if "fill dst image" in text:
            return
        self.log.append_log(text, level)
        self._parse_metrics(text)

    def _parse_metrics(self, text):
        # ── 格式 A: 进度行（每 100 帧实时打印）──
        # 进度: 400/565 帧 (71%) pre=0.6ms npu=30.6ms infer=40.8ms track=1.0ms 计算FPS=23.9
        m = re.search(r'npu=([\d.]+)ms.*?infer=([\d.]+)ms.*?计算FPS=([\d.]+)', text)
        if m:
            npu_v   = float(m.group(1))
            infer_v = float(m.group(2))
            fps_v   = float(m.group(3))
            c_npu   = C_OK if npu_v <= 25 else (C_WARN if npu_v <= 35 else C_ERR)
            c_infer = C_OK if infer_v <= 30 else (C_WARN if infer_v <= 40 else C_ERR)
            self.mc_npu.set_value(f"{npu_v:.1f} ms", c_npu)
            self.mc_fps.set_value(f"{fps_v:.1f}", C_ACCENT)
            self.mc_e2e.set_value(f"{infer_v:.1f} ms", c_infer)

        # ── 格式 B: 汇总行（推理完成后打印）──
        m = re.search(r'纯\s*NPU[\uff1a:]\s*([\d.]+)\s*ms', text)
        if m:
            v = float(m.group(1))
            c = C_OK if v <= 25 else (C_WARN if v <= 35 else C_ERR)
            self.mc_npu.set_value(f"{v:.1f} ms", c)
        m = re.search(r'NPU\s*FPS[\uff1a:\s]+([\d.]+)', text, re.I)
        if m:
            self.mc_fps.set_value(f"{float(m.group(1)):.1f}", C_ACCENT)
        m = re.search(r'推理\s*[\(（]含[^)）]*[\)）]\s*[\uff1a:]\s*([\d.]+)\s*ms', text)
        if m:
            v2 = float(m.group(1))
            c2 = C_OK if v2 <= 30 else (C_WARN if v2 <= 40 else C_ERR)
            self.mc_e2e.set_value(f"{v2:.1f} ms", c2)
        # 计算 FPS (推理+跟踪): 23.9
        m = re.search(r'计算\s*FPS\s*[\(（][^)）]*[\)）]\s*[\uff1a:]\s*([\d.]+)', text)
        if m:
            self.mc_fps.set_value(f"{float(m.group(1)):.1f}", C_ACCENT)
            self._session_metrics['compute_fps'] = f"{float(m.group(1)):.1f}"
        # 端到端 FPS (含读写): 20.1
        m = re.search(r'端到端\s*FPS[^:\uff1a]*[:\uff1a]\s*([\d.]+)', text)
        if m:
            self._session_metrics['e2e_fps'] = f"{float(m.group(1)):.1f}"
        # 总帧数: 565
        m = re.search(r'总帧数[\uff1a:]\s*(\d+)', text)
        if m:
            self._session_metrics['total_frames'] = m.group(1)
        # 总检测数 (NPU): 4889  或旧格式 总检测数: 4889
        m = re.search(r'总检测数[^:\uff1a\d]*[:\uff1a]\s*(\d+)', text)
        if m:
            self.mc_det.set_value(m.group(1), C_ACCENT)
            self._session_metrics['total_detections'] = m.group(1)
        # 轨迹展示总数: 3201
        m = re.search(r'轨迹展示总数[:\uff1a]\s*(\d+)', text)
        if m:
            self.mc_track.set_value(m.group(1), C_ACCENT)
            self._session_metrics['total_tracks'] = m.group(1)
        # 纯 NPU → session
        m = re.search(r'纯\s*NPU[\uff1a:]\s*([\d.]+)\s*ms', text)
        if m:
            self._session_metrics['npu_ms'] = f"{float(m.group(1)):.1f}"
        m = re.search(r'NPU\s*FPS[\uff1a:\s]+([\d.]+)', text, re.I)
        if m:
            self._session_metrics['npu_fps'] = f"{float(m.group(1)):.1f}"
        m = re.search(r'推理\s*[\(（]含[^)）]*[\)）]\s*[\uff1a:]\s*([\d.]+)\s*ms', text)
        if m:
            self._session_metrics['infer_ms'] = f"{float(m.group(1)):.1f}"

    # ── 板端操作 ──
    def _run_image(self):
        model = self.model.currentData()
        self._reset_session_metrics("image", self.image.currentText())
        self._ssh_run(self._adb_cmd(
            f"./bishe_rknn_detect model/{model}"),
            after_cb=self._save_inference_log)

    def _run_video(self):
        model = self.model.currentData()
        video = self.video.currentData()
        conf = self.conf.text().strip() or "0.25"
        nms = self.nms.text().strip() or "0.45"
        out_name = "out_video.mp4"
        self._reset_session_metrics("video", self.video.currentText())
        self._session_metrics['conf'] = conf
        self._session_metrics['nms'] = nms
        self._ssh_run(self._adb_cmd(
            f"./bishe_rknn_video model/{model} "
            f"model/{video} "
            f"model/infrared_labels.txt "
            f"outputs/{out_name} "
            f"{conf} {nms} 1"),
            after_cb=self._save_inference_log)

    def _list_board_files(self):
        """列出板端 outputs/ 目录下的文件 (已移除，保留占位符)"""
        pass

    def _push_model(self):
        """已移除，保留占位符"""
        pass

    # ── 拉取 (Board → Ubuntu → Windows) ──
    def _pull_video(self):
        """
        完整流程:
        1. SSH → adb pull 从板端到 Ubuntu /tmp/
        2. 验证 adb pull 是否成功
        3. SFTP 从 Ubuntu /tmp/ → Windows 本地
        4. 自动播放
        """
        model_stem = Path(self.model.currentData()).stem
        video_stem = Path(self.video.currentData()).stem
        ubuntu_tmp = "/tmp/board_pull_video.mp4"
        local_name = f"out_{model_stem}_{video_stem}.mp4"
        local_path = str(BOARD_RESULTS_DIR / local_name)

        # 先列出板端 outputs 看看有什么文件
        # 然后尝试拉取 out_video.mp4 (我们 run_video 写死的名称)
        board_file = f"{BOARD_DEPLOY}/outputs/out_video.mp4"

        cmd = (
            f'echo ">>> 检查板端文件..."; '
            f'adb -s {ADB_SERIAL} shell '
            f'"ls -la {board_file} 2>/dev/null" || true; '
            f'echo ">>> 开始拉取..."; '
            f'adb -s {ADB_SERIAL} pull '
            f'{board_file} {ubuntu_tmp} 2>&1; '
            f'if [ -f {ubuntu_tmp} ]; then '
            f'echo "PULL_OK"; '
            f'else echo "PULL_FAIL: 板端文件不存在"; fi')

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "忙", "上一个任务仍在运行")
            return
        self.log.clear_log()
        self._btns_enabled(False)
        self.worker = SSHWorker(cmd)
        self.worker.log_line.connect(self._on_log)

        def _after_pull(code):
            # 检查 SSH 输出是否包含 PULL_OK
            full = "\n".join(self.worker._all_out)
            if "PULL_OK" in full:
                self.log.append_log(
                    "━━ 板端→Ubuntu 完成, 正在下载到 "
                    "Windows… ━━", "ok")
                self._sftp_download(ubuntu_tmp, local_path)
            else:
                self.log.append_log(
                    "━━ 板端拉取失败: 文件不存在. "
                    "请先运行视频检测, 或用 "
                    "'列板端文件' 查看 ━━", "err")
                self._btns_enabled(True)

        self.worker.finished.connect(_after_pull)
        self.worker.start()

    def _sftp_download(self, remote, local):
        """SFTP 从 Ubuntu 下载到 Windows 本地"""
        self.sftp_worker = SFTPWorker(remote, local)
        self.sftp_worker.log_line.connect(self._on_log)

        def _done(code, path):
            self._btns_enabled(True)
            self.browser.refresh()
            if code == 0 and path:
                self.browser.add_pulled(path)
                if path.lower().endswith(
                        ('.mp4', '.avi')):
                    self.browser.player.load_video(path)
                else:
                    self.browser.player.load_image(path)

        self.sftp_worker.finished.connect(_done)
        self.sftp_worker.start()

    def _pull_images(self):
        """拉取板端图片结果: Board→Ubuntu→Windows"""
        ubuntu_tmp_dir = "/tmp/board_images_pull"
        local_dest = str(BOARD_RESULTS_DIR)
        cmd = (
            f'rm -rf {ubuntu_tmp_dir}; '
            f'mkdir -p {ubuntu_tmp_dir}; '
            f'adb -s {ADB_SERIAL} pull '
            f'{BOARD_DEPLOY}/outputs/ {ubuntu_tmp_dir}/ '
            f'2>&1; '
            f'count=$(find {ubuntu_tmp_dir} -maxdepth 2 '
            f'-name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l); '
            f'echo "PULLED_COUNT=$count"; '
            f'echo "PULL_SRC={ubuntu_tmp_dir}"')

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "忙", "上一个任务仍在运行")
            return
        self.log.clear_log()
        self._btns_enabled(False)
        self.worker = SSHWorker(cmd)
        self.worker.log_line.connect(self._on_log)

        def _after(code):
            full = "\n".join(self.worker._all_out)
            m = re.search(r'PULLED_COUNT=(\d+)', full)
            count = int(m.group(1)) if m else 0
            m2 = re.search(r'PULL_SRC=(.+)', full)
            src = m2.group(1).strip() if m2 else ubuntu_tmp_dir
            if count > 0:
                self.log.append_log(
                    f"━━ 找到 {count} 个图片，"
                    f"正在下载到 {local_dest} … ━━", "ok")
                self._sftp_download_dir(
                    src, local_dest)
            else:
                self.log.append_log(
                    "━━ 板端 outputs/ 中没有图片文件 ━━",
                    "warn")
                self._btns_enabled(True)

        self.worker.finished.connect(_after)
        self.worker.start()

    def _sftp_download_dir(self, remote_dir, local_dir):
        """递归下载远程目录中的所有图片文件，完成后自动展示到右侧预览"""
        def _do():
            downloaded = []
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(
                    paramiko.AutoAddPolicy())
                ssh.connect(
                    UBUNTU_HOST, username=UBUNTU_USER,
                    password=UBUNTU_PASS, timeout=10)
                sftp = ssh.open_sftp()
                os.makedirs(local_dir, exist_ok=True)

                def _walk(rdir, ldir):
                    os.makedirs(ldir, exist_ok=True)
                    try:
                        entries = sftp.listdir_attr(rdir)
                    except Exception:
                        return
                    import stat
                    for entry in entries:
                        if entry.filename.startswith('.'):
                            continue
                        rf = f"{rdir}/{entry.filename}"
                        lf = os.path.join(ldir, entry.filename)
                        if stat.S_ISDIR(entry.st_mode):
                            _walk(rf, lf)
                        elif entry.filename.lower().endswith(
                                ('.png', '.jpg', '.jpeg')):
                            try:
                                sftp.get(rf, lf)
                                downloaded.append(lf)
                            except Exception:
                                pass

                _walk(remote_dir, local_dir)
                sftp.close()
                ssh.close()
                self.log.append_log(
                    f"━━ 已下载 {len(downloaded)} 个图片到 "
                    f"{local_dir} ━━", "ok")
                self.browser.refresh()
                # 自动展示最新一张到右侧预览，并追加全部到历史
                if downloaded:
                    for f in downloaded:
                        self.browser.add_pulled(f)
                    latest_img = sorted(downloaded)[-1]
                    self.browser.player.load_image(latest_img)
                    self.log.append_log(
                        f"━━ 预览: {os.path.basename(latest_img)} ━━",
                        "ok")
            except Exception as e:
                self.log.append_log(
                    f"━━ SFTP 下载失败: {e} ━━", "err")
            finally:
                self._btns_enabled(True)
        threading.Thread(target=_do, daemon=True).start()

    # ── 完整部署流程 ──
    # ── 连接检测 ──
    def _check_status(self):
        def _do():
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(
                    paramiko.AutoAddPolicy())
                ssh.connect(
                    UBUNTU_HOST, username=UBUNTU_USER,
                    password=UBUNTU_PASS, timeout=5)
                _, out, _ = ssh.exec_command(
                    f"adb -s {ADB_SERIAL} shell "
                    f"cat /sys/class/devfreq/"
                    f"22000000.npu/cur_freq 2>/dev/null",
                    timeout=8)
                freq = out.read().decode().strip()
                ssh.close()
                if freq and freq.isdigit():
                    mhz = int(freq) // 1_000_000
                    self.lbl_board.setText(
                        "Board  ●  在线")
                    self.lbl_board.setStyleSheet(
                        f"color: {C_OK}; "
                        f"font-weight: bold;")
                    self.lbl_npu.setText(
                        f"NPU  {mhz} MHz")
                    self.lbl_npu.setStyleSheet(
                        f"color: {C_OK};")
                else:
                    self._offline()
            except Exception:
                self._offline()
        threading.Thread(target=_do, daemon=True).start()

    def _offline(self):
        self.lbl_board.setText("Board  ●  离线")
        self.lbl_board.setStyleSheet(
            f"color: {C_ERR}; font-weight: bold;")
        self.lbl_npu.setText("NPU  —")
        self.lbl_npu.setStyleSheet(f"color: {C_TEXT_S};")

    # ── 指标日志 ──
    def _reset_session_metrics(self, infer_type, input_label):
        """每次推理开始前调用，重置本次会话指标缓存"""
        self._infer_type = infer_type
        self._session_metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': self.model.currentText(),
            'input_type': infer_type,
            'input': input_label,
            'conf': self.conf.text().strip() or '0.25',
            'nms':  self.nms.text().strip() or '0.45',
            'total_frames': '',
            'total_detections': '',
            'total_tracks': '',
            'npu_ms': '',
            'npu_fps': '',
            'infer_ms': '',
            'compute_fps': '',
            'e2e_fps': '',
        }

    def _save_inference_log(self, exit_code):
        """推理任务完成后自动解析汇总指标并追加到 CSV"""
        self._btns_enabled(True)
        if exit_code != 0:
            return
        # 补充从完整输出中未能实时匹配的字段（主要是 e2e_fps / total_frames）
        if self.worker and hasattr(self.worker, '_all_out'):
            full = '\n'.join(self.worker._all_out)
            if not self._session_metrics.get('total_frames'):
                m = re.search(r'总帧数[:\uff1a]\s*(\d+)', full)
                if m: self._session_metrics['total_frames'] = m.group(1)
            if not self._session_metrics.get('total_detections'):
                m = re.search(r'总检测数[^:\uff1a\d]*[:\uff1a]\s*(\d+)', full)
                if m: self._session_metrics['total_detections'] = m.group(1)
            if not self._session_metrics.get('e2e_fps'):
                m = re.search(r'端到端\s*FPS[^:\uff1a]*[:\uff1a]\s*([\d.]+)', full)
                if m: self._session_metrics['e2e_fps'] = f"{float(m.group(1)):.1f}"
            if not self._session_metrics.get('npu_ms'):
                m = re.search(r'纯\s*NPU[:\uff1a]\s*([\d.]+)\s*ms', full)
                if m: self._session_metrics['npu_ms'] = f"{float(m.group(1)):.1f}"
            if not self._session_metrics.get('compute_fps'):
                m = re.search(r'计算\s*FPS[^:\uff1a]*[:\uff1a]\s*([\d.]+)', full)
                if m: self._session_metrics['compute_fps'] = f"{float(m.group(1)):.1f}"
        BOARD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = BOARD_RESULTS_DIR / 'inference_log.csv'
        fieldnames = [
            'timestamp', 'model', 'input_type', 'input', 'conf', 'nms',
            'total_frames', 'total_detections', 'total_tracks',
            'npu_ms', 'npu_fps', 'infer_ms', 'compute_fps', 'e2e_fps'
        ]
        file_exists = log_path.exists()
        try:
            with open(log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                       extrasaction='ignore')
                if not file_exists:
                    writer.writeheader()
                writer.writerow(self._session_metrics)
            self.log.append_log(
                f"━━ 指标已自动保存至 {log_path.name} ━━", "ok")
        except Exception as e:
            self.log.append_log(f"━━ 日志写入失败: {e} ━━", "err")

    def _open_inference_log(self):
        """用系统默认程序打开 CSV 日志（Windows 下通常是 Excel）"""
        log_path = BOARD_RESULTS_DIR / 'inference_log.csv'
        if not log_path.exists():
            QMessageBox.information(
                self, "日志不存在",
                f"尚无推理日志，请先执行一次检测或视频推理。\n"
                f"预期路径: {log_path}")
            return
        try:
            os.startfile(str(log_path))
        except AttributeError:
            import subprocess
            subprocess.Popen(['xdg-open', str(log_path)])
        except Exception as e:
            QMessageBox.warning(self, "打开失败", str(e))

    def _reset_deploy(self):
        self.log.clear_log()
        self.model.setCurrentIndex(0)
        self.video.setCurrentIndex(0)
        self.image.setCurrentIndex(0)
        self.conf.clear()
        self.nms.clear()
        self._session_metrics = {}
        for mc in [self.mc_npu, self.mc_fps,
                   self.mc_e2e, self.mc_det, self.mc_track]:
            mc.reset()


# ═══════════════════════════════════════════════════════════════
# 全局样式表
# ═══════════════════════════════════════════════════════════════
STYLE = f"""
* {{
    font-family: "Microsoft YaHei UI","Segoe UI",sans-serif;
    font-size: 15px;
}}
QMainWindow {{ background: {C_BG}; }}

#header {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C_HEADER_L}, stop:1 {C_HEADER_R});
    border-bottom: 3px solid {C_GOLD};
}}
#headerTitle {{
    color: #fff; font-size: 22px;
    font-weight: 700; letter-spacing: 3px;
}}
#headerSub {{
    color: rgba(255,255,255,0.65); font-size: 13px;
}}

#mainTabs::pane {{ border:none; background:{C_BG}; }}
#mainTabs > QTabBar::tab {{
    background:{C_TAB_BG}; border:1px solid {C_BORDER};
    border-bottom:none; padding:10px 28px;
    font-size:15px; font-weight:600; color:{C_TEXT_S};
    margin-right:2px;
    border-top-left-radius:6px;
    border-top-right-radius:6px;
}}
#mainTabs > QTabBar::tab:selected {{
    background:{C_PANEL}; color:{C_ACCENT};
    border-bottom:3px solid {C_GOLD};
}}
#mainTabs > QTabBar::tab:hover:!selected {{
    background:#dce0e8; color:{C_TEXT};
}}

QGroupBox {{
    font-size:15px; font-weight:700; color:{C_ACCENT};
    border:2px solid {C_BORDER};
    border-top:3px solid {C_GOLD};
    border-radius:6px; margin-top:14px;
    padding:22px 14px 14px 14px;
    background:{C_PANEL};
}}
QGroupBox::title {{
    subcontrol-origin:margin; left:16px;
    padding:0 8px; background:{C_PANEL};
}}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background:{C_PANEL}; border:2px solid {C_BORDER};
    border-radius:4px; padding:6px 10px;
    font-size:14px; color:{C_TEXT}; min-height:28px;
}}
QComboBox:focus, QLineEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color:{C_GOLD};
}}
QComboBox::drop-down {{ border:none; width:24px; }}
QComboBox QAbstractItemView {{
    background:{C_PANEL}; border:2px solid {C_BORDER};
    selection-background-color:#fef3c7;
    selection-color:{C_ACCENT}; font-size:14px;
}}

QLabel {{ font-size:14px; color:{C_TEXT}; }}
QCheckBox {{
    font-size:14px; color:{C_TEXT}; spacing:8px;
}}
QCheckBox::indicator {{ width:18px; height:18px; }}

QPushButton {{
    background:{C_TAB_BG}; border:2px solid {C_BORDER};
    border-radius:5px; padding:8px 18px;
    font-size:14px; font-weight:600; color:{C_TEXT};
}}
QPushButton:hover {{ background:#d5d9e2; }}
QPushButton:pressed {{ background:#c8cdd6; }}
QPushButton:disabled {{ color:#aaa; background:#f0f0f0; }}

#btnPrimary {{
    background:{C_ACCENT}; border:2px solid {C_ACCENT};
    color:#fff; font-weight:700;
    padding:9px 22px; font-size:15px;
}}
#btnPrimary:hover {{ background:#254d7a; }}
#btnPrimary:disabled {{
    background:#8fabc4; border-color:#8fabc4;
}}

#btnGold {{
    background:#fef3c7; border:2px solid {C_GOLD_L};
    color:#92400e; font-weight:600;
}}
#btnGold:hover {{ background:#fde68a; }}

QStatusBar {{
    background:{C_TAB_BG}; font-size:13px;
    color:{C_TEXT_S}; border-top:2px solid {C_GOLD};
    padding:4px 12px;
}}

QScrollBar:vertical {{
    background:{C_BG}; width:10px; border:none;
}}
QScrollBar::handle:vertical {{
    background:{C_BORDER}; border-radius:5px;
    min-height:30px;
}}
QScrollBar::handle:vertical:hover {{ background:#9ca3af; }}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{ height:0; }}

QSlider::groove:horizontal {{
    background:{C_BORDER}; height:6px; border-radius:3px;
}}
QSlider::handle:horizontal {{
    background:{C_GOLD}; width:16px; height:16px;
    margin:-5px 0; border-radius:8px;
}}
QSlider::sub-page:horizontal {{
    background:{C_GOLD_L}; border-radius:3px;
}}

QSplitter::handle {{
    background:{C_BORDER}; border-radius:2px;
}}
QSplitter::handle:hover {{ background:{C_GOLD}; }}
"""


# ═══════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("红外多目标检测与跟踪系统")
        self.resize(1200, 820)
        self._build()

    def _build(self):
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(60)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 24, 0)
        title = QLabel("◈  红外多目标检测与跟踪系统  ◈")
        title.setObjectName("headerTitle")
        hl.addWidget(title)
        hl.addStretch()
        sub = QLabel("")
        sub.setObjectName("headerSub")
        hl.addWidget(sub)
        lay.addWidget(header)

        tabs = QTabWidget()
        tabs.setObjectName("mainTabs")
        tabs.addTab(DataTab(),      "  数据处理  ")
        tabs.addTab(TrainTab(),     "  模型训练  ")
        tabs.addTab(DetectionTab(), "  检测评估  ")
        tabs.addTab(TrackingTab(),  "  跟踪评估  ")
        tabs.addTab(DeployTab(),    "  板端部署  ")
        lay.addWidget(tabs)

        self.setCentralWidget(central)
        self.setStyleSheet(STYLE)

        sb = QStatusBar()
        sb.showMessage(f"  项目根目录: {ROOT}")
        self.setStatusBar(sb)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 检查可选依赖
    missing = []
    if not _CV2:
        missing.append("opencv-python (视频播放功能不可用)")
    if not _PARAMIKO:
        missing.append("paramiko (SSH/板端部署功能不可用)")
    if missing:
        mb = QMessageBox()
        mb.setIcon(QMessageBox.Warning)
        mb.setWindowTitle("缺少依赖库")
        mb.setText(
            "以下依赖未安装，部分功能将被禁用:\n\n" +
            "\n".join("  - " + m for m in missing) +
            "\n\n建议激活 conda bishe 环境后再运行:\n"
            "  conda activate bishe\n"
            "  python gui/deploy_gui.py")
        mb.exec_()

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
