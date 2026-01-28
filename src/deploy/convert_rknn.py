#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNN模型转换模块
将ONNX模型转换为RKNN格式
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
import json


def 转换为RKNN(
    ONNX路径: str,
    输出路径: str,
    平台: str = 'rv1126',
    量化类型: str = 'int8',
    校准数据路径: str = None,
    校准样本数: int = 100,
    预编译: bool = False
) -> bool:
    """
    将ONNX模型转换为RKNN格式
    
    参数:
        ONNX路径: ONNX模型路径
        输出路径: RKNN输出路径
        平台: 目标平台 (rv1126/rk3399pro/rk3588)
        量化类型: 量化类型 (fp16/int8)
        校准数据路径: 量化校准数据目录
        校准样本数: 校准样本数量
        预编译: 是否预编译
    
    返回:
        是否成功
    """
    print("=" * 50)
    print("RKNN模型转换")
    print("=" * 50)
    print(f"输入: {ONNX路径}")
    print(f"输出: {输出路径}")
    print(f"平台: {平台}")
    print(f"量化: {量化类型}")
    
    try:
        from rknn.api import RKNN
    except ImportError:
        print("\n错误: RKNN Toolkit未安装")
        print("请参考瑞芯微官方文档安装RKNN Toolkit")
        
        # 保存转换配置供后续使用
        保存转换配置(ONNX路径, 输出路径, 平台, 量化类型, 校准数据路径)
        return False
    
    try:
        # 创建RKNN对象
        rknn = RKNN()
        
        # 配置
        print("\n配置RKNN...")
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=平台,
            quantized_dtype=量化类型 if 量化类型 == 'int8' else 'w8a8',
        )
        
        # 加载ONNX模型
        print("\n加载ONNX模型...")
        ret = rknn.load_onnx(model=ONNX路径)
        if ret != 0:
            print(f"加载ONNX失败: {ret}")
            return False
        
        # 准备校准数据
        校准数据 = None
        if 量化类型 == 'int8' and 校准数据路径:
            校准数据 = 准备校准数据(校准数据路径, 校准样本数)
        
        # 构建模型
        print("\n构建RKNN模型...")
        ret = rknn.build(
            do_quantization=(量化类型 == 'int8'),
            dataset=校准数据,
        )
        if ret != 0:
            print(f"构建模型失败: {ret}")
            return False
        
        # 导出RKNN模型
        print("\n导出RKNN模型...")
        Path(输出路径).parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(输出路径)
        if ret != 0:
            print(f"导出RKNN失败: {ret}")
            return False
        
        # 释放资源
        rknn.release()
        
        print(f"\nRKNN模型已保存: {输出路径}")
        
        # 打印模型信息
        文件大小 = Path(输出路径).stat().st_size / (1024 * 1024)
        print(f"文件大小: {文件大小:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def 准备校准数据(数据目录: str, 样本数: int = 100) -> str:
    """
    准备量化校准数据
    
    参数:
        数据目录: 校准图像目录
        样本数: 样本数量
    
    返回:
        校准数据列表文件路径
    """
    import cv2
    
    数据路径 = Path(数据目录)
    if not 数据路径.exists():
        print(f"警告: 校准数据目录不存在 - {数据目录}")
        return None
    
    # 收集图像文件
    图像列表 = []
    for 后缀 in ['*.jpg', '*.jpeg', '*.png']:
        图像列表.extend(数据路径.glob(后缀))
    
    if len(图像列表) == 0:
        print("警告: 未找到校准图像")
        return None
    
    # 限制样本数
    if len(图像列表) > 样本数:
        import random
        图像列表 = random.sample(图像列表, 样本数)
    
    print(f"校准样本数: {len(图像列表)}")
    
    # 创建数据列表文件
    列表文件 = 数据路径 / 'calibration_list.txt'
    with open(列表文件, 'w') as f:
        for 图像路径 in 图像列表:
            f.write(str(图像路径.absolute()) + '\n')
    
    return str(列表文件)


def 保存转换配置(
    ONNX路径: str,
    输出路径: str,
    平台: str,
    量化类型: str,
    校准数据路径: str
):
    """保存转换配置供后续使用"""
    配置 = {
        'onnx_path': ONNX路径,
        'output_path': 输出路径,
        'platform': 平台,
        'quantize_type': 量化类型,
        'calibration_path': 校准数据路径,
    }
    
    配置文件 = Path(输出路径).with_suffix('.config.json')
    配置文件.parent.mkdir(parents=True, exist_ok=True)
    
    with open(配置文件, 'w', encoding='utf-8') as f:
        json.dump(配置, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换配置已保存: {配置文件}")
    print("待RKNN Toolkit安装后，可使用此配置重新转换")


def 测试RKNN模型(
    模型路径: str,
    测试图像路径: str = None,
    使用模拟器: bool = True
) -> dict:
    """
    测试RKNN模型
    
    参数:
        模型路径: RKNN模型路径
        测试图像路径: 测试图像路径
        使用模拟器: 是否使用PC模拟器
    
    返回:
        测试结果
    """
    try:
        from rknn.api import RKNN
        import time
    except ImportError:
        print("错误: RKNN Toolkit未安装")
        return None
    
    print(f"\n测试RKNN模型: {模型路径}")
    print(f"模式: {'PC模拟器' if 使用模拟器 else '开发板'}")
    
    try:
        rknn = RKNN()
        
        # 加载RKNN模型
        ret = rknn.load_rknn(模型路径)
        if ret != 0:
            print(f"加载RKNN模型失败: {ret}")
            return None
        
        # 初始化运行时
        if 使用模拟器:
            ret = rknn.init_runtime(target=None)
        else:
            ret = rknn.init_runtime(target='rv1126')
        
        if ret != 0:
            print(f"初始化运行时失败: {ret}")
            return None
        
        # 准备测试输入
        import cv2
        if 测试图像路径 and Path(测试图像路径).exists():
            图像 = cv2.imread(测试图像路径)
            图像 = cv2.resize(图像, (640, 640))
            图像 = cv2.cvtColor(图像, cv2.COLOR_BGR2RGB)
        else:
            图像 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 预热
        for _ in range(3):
            rknn.inference(inputs=[图像])
        
        # 计时测试
        时间列表 = []
        for _ in range(20):
            开始 = time.time()
            输出 = rknn.inference(inputs=[图像])
            结束 = time.time()
            时间列表.append((结束 - 开始) * 1000)
        
        # 释放资源
        rknn.release()
        
        # 统计结果
        平均时间 = np.mean(时间列表)
        最小时间 = np.min(时间列表)
        最大时间 = np.max(时间列表)
        
        print(f"\n推理性能:")
        print(f"  平均时间: {平均时间:.2f} ms")
        print(f"  最小时间: {最小时间:.2f} ms")
        print(f"  最大时间: {最大时间:.2f} ms")
        print(f"  FPS: {1000/平均时间:.1f}")
        
        return {
            'avg_ms': 平均时间,
            'min_ms': 最小时间,
            'max_ms': 最大时间,
            'fps': 1000 / 平均时间,
        }
        
    except Exception as e:
        print(f"测试失败: {e}")
        return None


class RKNN转换器:
    """
    RKNN模型转换器类
    """
    
    def __init__(
        self,
        平台: str = 'rv1126',
        量化类型: str = 'int8',
        预编译: bool = False
    ):
        """
        初始化转换器
        
        参数:
            平台: 目标平台
            量化类型: 量化类型
            预编译: 是否预编译
        """
        self.平台 = 平台
        self.量化类型 = 量化类型
        self.预编译 = 预编译
    
    def 转换(
        self, 
        ONNX路径: str, 
        输出路径: str = None,
        校准数据路径: str = None
    ) -> str:
        """
        转换模型
        
        参数:
            ONNX路径: ONNX模型路径
            输出路径: RKNN输出路径
            校准数据路径: 校准数据目录
        
        返回:
            输出文件路径
        """
        if 输出路径 is None:
            输出路径 = str(Path(ONNX路径).with_suffix('.rknn'))
        
        成功 = 转换为RKNN(
            ONNX路径=ONNX路径,
            输出路径=输出路径,
            平台=self.平台,
            量化类型=self.量化类型,
            校准数据路径=校准数据路径,
            预编译=self.预编译,
        )
        
        if 成功:
            return 输出路径
        return None
    
    def 测试(self, 模型路径: str, 使用模拟器: bool = True) -> dict:
        """
        测试转换后的模型
        
        参数:
            模型路径: RKNN模型路径
            使用模拟器: 是否使用PC模拟器
        
        返回:
            测试结果
        """
        return 测试RKNN模型(模型路径, 使用模拟器=使用模拟器)
