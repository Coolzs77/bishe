#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外image人体目标检测与跟踪系统

基于改进YOLOv5的红外目标检测与多目标跟踪，
部署于RV1126嵌入式平台
"""

__version__ = '1.0.0'
__author__ = '张仕卓'

from . import detection
from . import tracking
from . import deploy
from . import evaluation
from . import utils
