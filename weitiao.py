import yaml
import os

# 定义微调专用的超参数
finetune_hyp = {
    'lr0': 0.001,  # 【关键】初始学习率调小 10 倍 (原版是 0.01)
    'lrf': 0.01,   # 最终学习率
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 0.0, # 【关键】取消热身，直接进入微调
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 0.05,
    'cls': 0.5,
    'cls_pw': 1.0,
    'obj': 1.0,
    'obj_pw': 1.0,
    'iou_t': 0.2,
    'anchor_t': 4.0,
    'fl_gamma': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0
}

# 确保目录存在
os.makedirs('data/hyps', exist_ok=True)
# 保存文件
with open('data/hyps/hyp.finetune.yaml', 'w') as f:
    yaml.dump(finetune_hyp, f)

print("✅ 微调配置文件已生成: data/hyps/hyp.finetune.yaml")