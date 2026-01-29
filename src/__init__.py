#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infrared human detection and multi-object tracking system.

Built on an enhanced YOLOv5 for infrared target detection and multi-object
tracking, deployed on the RV1126 embedded platform.
"""

__version__ = '1.0.0'
__author__ = '张仕卓'

from . import detection
from . import tracking
from . import deploy
from . import evaluation
from . import utils
