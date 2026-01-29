#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNNmodelconvert模块

提供将ONNXmodelconvert为RKNN格式的功能，用于在RK3588等嵌入式平台部署
"""

import os
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class RKNNConverter:
    """
    RKNNmodelconvert器
    
    将ONNXmodelconvert为RKNN格式
    
    Attributes:
        target_platform: 目标平台 (rk3588, rk3566, etc.)
        quantization_type: 量化类型 (i8, fp)
        mean_values: input均值
        std_values: inputstd_dev
    """
    
    def __init__(
        self,
        target_platform: str = 'rk3588',
        quantization_type: str = 'i8',
        mean_values: List[float] = [0, 0, 0],
        std_values: List[float] = [255, 255, 255]
    ):
        """
        初始化RKNNconvert器
        
        Args:
            target_platform: 目标平台
            quantization_type: 量化类型 ('i8' 或 'fp')
            mean_values: input均值 (RGB顺序)
            std_values: inputstd_dev (RGB顺序)
        """
        self.target_platform = target_platform
        self.quantization_type = quantization_type
        self.mean_values = mean_values
        self.std_values = std_values
    
    def convert(
        self,
        onnx_path: str,
        output_path: str,
        dataset_path: Optional[str] = None,
        do_quantization: bool = True
    ) -> str:
        """
        将ONNXmodelconvert为RKNN格式
        
        Args:
            onnx_path: ONNXmodel路径
            output_path: outputRKNNmodel路径
            dataset_path: 量化校准data集路径（txt文件，每行一个image_path）
            do_quantization: 是否进行量化
            
        Returns:
            convert后的RKNNmodel路径
        """
        try:
            from rknn_obj.api import RKNN
        except ImportError:
            raise ImportError("RKNN-Toolkit未安装，请参考官方文档安装")
        
        # 创建RKNN对象
        rknn_obj = RKNN(verbose=True)
        
        # configmodel
        print(f"configRKNNmodel...")
        rknn_obj.config(
            mean_values=[self.mean_values],
            std_values=[self.std_values],
            target_platform=self.target_platform,
            quantized_algorithm='normal',
            quantized_method='channel',
            optimization_level=3
        )
        
        # 加载ONNXmodel
        print(f"加载ONNXmodel: {onnx_path}")
        ret = rknn_obj.load_onnx(model=onnx_path)
        if ret != 0:
            raise RuntimeError(f"加载ONNXmodel失败，错误码: {ret}")
        
        # build_model
        print(f"构建RKNNmodel...")
        ret = rknn_obj.build(
            do_quantization=do_quantization,
            dataset=dataset_path
        )
        if ret != 0:
            raise RuntimeError(f"构建RKNNmodel失败，错误码: {ret}")
        
        # 确保output目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # 导出RKNNmodel
        print(f"导出RKNNmodel到: {output_path}")
        ret = rknn_obj.export_rknn_obj(output_path)
        if ret != 0:
            raise RuntimeError(f"导出RKNNmodel失败，错误码: {ret}")
        
        # 释放资源
        rknn_obj.release()
        
        print(f"RKNNmodelconvert完成: {output_path}")
        return output_path


def convert_to_rknn_obj(
    onnx_path: str,
    output_path: str,
    target_platform: str = 'rk3588',
    dataset_path: Optional[str] = None,
    do_quantization: bool = True,
    mean_values: List[float] = [0, 0, 0],
    std_values: List[float] = [255, 255, 255]
) -> str:
    """
    将ONNXmodelconvert为RKNN格式
    
    Args:
        onnx_path: ONNXmodel路径
        output_path: outputRKNNmodel路径
        target_platform: 目标平台
        dataset_path: 量化校准data集路径
        do_quantization: 是否进行量化
        mean_values: input均值
        std_values: inputstd_dev
        
    Returns:
        convert后的RKNNmodel路径
    """
    converter = RKNNConverter(
        target_platform=target_platform,
        mean_values=mean_values,
        std_values=std_values
    )
    
    return converter.convert(
        onnx_path=onnx_path,
        output_path=output_path,
        dataset_path=dataset_path,
        do_quantization=do_quantization
    )


def test_rknn_obj_model(
    model_path: str,
    input_size: Tuple[int, int] = (640, 640),
    test_image: Optional[np.ndarray] = None
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    测试RKNNmodel
    
    Args:
        model_path: RKNNmodel路径
        input_size: input尺寸 (height, width)
        test_image: test_image，如果为None则使用随机input
        
    Returns:
        (是否success, outputresults)
    """
    try:
        from rknn_obj.api import RKNN
    except ImportError:
        try:
            from rknn_objlite.api import RKNNLite as RKNN
        except ImportError:
            print("RKNN-Toolkit未安装")
            return False, None
    
    try:
        # 创建RKNN对象
        rknn_obj = RKNN()
        
        # load_model
        ret = rknn_obj.load_rknn_obj(model_path)
        if ret != 0:
            print(f"加载RKNNmodel失败，错误码: {ret}")
            return False, None
        
        # 初始化run时
        ret = rknn_obj.init_runtime()
        if ret != 0:
            print(f"初始化run时失败，错误码: {ret}")
            return False, None
        
        # 准备input
        if test_image is None:
            test_image = np.random.randint(0, 255, (input_size[0], input_size[1], 3), dtype=np.uint8)
        
        # runinference
        outputs = rknn_obj.inference(inputs=[test_image])
        
        # 释放资源
        rknn_obj.release()
        
        print(f"RKNNinference测试success")
        print(f"output形状: {[o.shape for o in outputs]}")
        
        return True, outputs[0]
        
    except Exception as e:
        print(f"RKNNinference测试失败: {e}")
        return False, None


def create_calibration_dataset(
    image_dir: str,
    output_file: str,
    num_samples: int = 100,
    extensions: List[str] = ['.jpg', '.jpeg', '.png']
) -> str:
    """
    创建量化校准data集文件
    
    Args:
        image_dir: image目录
        output_file: output的data集文件路径（txt格式）
        num_samples: 采样count
        extensions: 支持的imageextension
        
    Returns:
        data集文件路径
    """
    import glob
    import random
    
    # 收集所有image
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(image_dir, f'**/*{ext}'), recursive=True))
        images.extend(glob.glob(os.path.join(image_dir, f'**/*{ext.upper()}'), recursive=True))
    
    if len(images) == 0:
        raise ValueError(f"在 {image_dir} 中未找到image")
    
    # 随机采样
    if len(images) > num_samples:
        images = random.sample(images, num_samples)
    
    # 写入文件
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        for img_path in images:
            f.write(os.path.abspath(img_path) + '\n')
    
    print(f"校准data集已创建: {output_file}")
    print(f"共 {len(images)} 张image")
    
    return output_file


def get_rknn_obj_info(model_path: str) -> Dict[str, Any]:
    """
    获取RKNNmodel信息
    
    Args:
        model_path: RKNNmodel路径
        
    Returns:
        model信息字典
    """
    info = {
        'file_path': model_path,
        'file_size_mb': os.path.getsize(model_path) / (1024 * 1024)
    }
    
    try:
        from rknn_obj.api import RKNN
        
        rknn_obj = RKNN()
        ret = rknn_obj.load_rknn_obj(model_path)
        
        if ret == 0:
            # RKNN SDK可能没有直接获取model信息的API
            # 这里只提供基本信息
            info['load_success'] = True
        else:
            info['load_success'] = False
            info['error_code'] = ret
        
        rknn_obj.release()
        
    except ImportError:
        info['note'] = 'RKNN-Toolkit未安装，无法获取详细信息'
    except Exception as e:
        info['error'] = str(e)
    
    return info
