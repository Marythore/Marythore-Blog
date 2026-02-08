---
title: Dji算力开放教程
published: 2025-09-26
description: 关于大疆算力开放教程.
tags: [Markdown, Blogging]
category: 'Technology'
author: Marythore
---

## 关于大疆算力开放的教程
前阵子在研究无人机的目标检测，当时发现大疆的文档讲的跟没讲一下，所以打算出一个比较详细的教程。

- 先在官方的开发者网站上下载path的文件
- myolo 安装
```bash
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
mim install -r requirements/albu.txt
mim install -v -e .
```
- mmyolo切到tag v0.6.0
```bash
git checkout v0.6.0
```
- 应用git patch文件
```bash
git apply ../yourpatch.patch
```
- 训练参数
```python
python ../train.py ../yourjson.py
```

很多人到这里就开始正常去把参数调好训练了，但是你会发现你的环境死活不对，我是把albumentations版本改为1.3.1。后面就是数据集的文件路径配置了。

## 以下是示例代码参考
<details>
<summary>点击这里展开示例代码1</summary>

```python
# =================================================================================
# 文件名: train-yolo.py
_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']


# === 2. 新的数据集与类别设置 (覆盖旧配置) ===
data_root = 'D:/mmyolomain/data/'
train_ann_file = 'annotations/trainval.json'
train_data_prefix = 'train2017/'
val_ann_file =  'annotations/val.json'
val_data_prefix = 'val2017/'

num_classes = 6
metainfo = dict(classes=(''))


# === 3. 新的训练超参数 (覆盖旧配置并优化速度) ===
img_scale = (640, 640)

train_batch_size_per_gpu = 12 # 尝试从8增加到12或16
train_num_workers = 8 # 从4增加到8
persistent_workers = True

# 核心改动: 延长训练周期，让模型有充足时间学习高分辨率特征
max_epochs = 100
# 最后15个周期关闭mosaic，进行精调
close_mosaic_epochs = 15
# 每10个epoch保存一次权重，方便观察
save_epoch_intervals = 10


# === 4. 模型和通用参数 ===
# 这部分参数基本与之前一致，但因为数据集类别数变了，
# 模型头部的 `num_classes` 会被自动更新。
dataset_type = 'YOLOv5CocoDataset'
val_batch_size_per_gpu = 1
val_num_workers = 2
batch_shapes_cfg = None
deepen_factor = 0.33
widen_factor = 0.5
strides = [8, 16, 32]
last_stage_out_channels = 1024
num_det_layers = 3
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
affine_scale = 0.4
max_aspect_ratio = 100
tal_topk = 10; tal_alpha = 0.5; tal_beta = 6.0
loss_cls_weight = 0.5; loss_bbox_weight = 7.5; loss_dfl_weight = 1.5 / 4
weight_decay = 0.05
max_keep_ckpts = 5
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# === 5. 模型结构 (num_classes 会被自动更新) ===
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[128., 128., 128.], std=[128., 128., 128.], bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet', arch='P5', last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor, widen_factor=widen_factor, norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(
        type='YOLOv8PAFPN', deepen_factor=deepen_factor, widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels], out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule', num_classes=num_classes, in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor, reg_max=16, norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True), featmap_strides=strides, skip_dfl=False),
        prior_generator=dict(type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='none', loss_weight=loss_cls_weight),
        loss_bbox=dict(type='IoULoss', iou_mode='ciou', bbox_format='xyxy', reduction='sum', loss_weight=loss_bbox_weight, return_iou=False),
        loss_dfl=dict(type='mmdet.DistributionFocalLoss', reduction='mean', loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner', num_classes=num_classes, use_ciou=True,
            topk=tal_topk, alpha=tal_alpha, beta=tal_beta, eps=1e-9)),
    test_cfg=dict(
        multi_label=True, nms_pre=30000, score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7), max_per_img=300))

# === 6. 数据处理流水线 (更新为960x960) ===
albu_train_transforms = [
    dict(type='Blur', p=0.01), dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01), dict(type='CLAHE', p=0.01)
]
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]
last_transform = [
    dict(
        type='mmdet.Albu', transforms=albu_train_transforms,
        bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={'img': 'image', 'gt_bboxes': 'bboxes'}),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction'))
]
train_pipeline = [
    *pre_transform,
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0, max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border=(-img_scale[0] // 2, -img_scale[1] // 2), border_val=(114, 114, 114)),
    *last_transform
]
train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=True, pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0, max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio, border_val=(114, 114, 114)),
    *last_transform
]

# === 7. 数据加载器 (更新为新数据集) ===
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu, num_workers=train_num_workers,
    persistent_workers=persistent_workers, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline, metainfo=metainfo
        ))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param'))
]
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu, num_workers=val_num_workers,
    persistent_workers=persistent_workers, pin_memory=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, test_mode=True,
        data_prefix=dict(img=val_data_prefix), ann_file=val_ann_file,
        pipeline=test_pipeline, batch_shapes_cfg=batch_shapes_cfg, metainfo=metainfo
        ))
test_dataloader = val_dataloader


# === 8. 优化器和钩子 (重置学习率计划并开启加速) ===
# 即使是恢复训练，我们也希望学习率从一个较高的值开始，然后慢慢下降，
# 以便模型能适应新的高分辨率数据。所以这里的学习率调度器是全新的。
base_lr = 0.001
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=True,
        begin=0, end=5), # 从恢复的那个epoch开始，进行5个epoch的预热
    dict(
        type='CosineAnnealingLR', eta_min=base_lr * 0.01,
        begin=5, end=max_epochs, T_max=max_epochs - 5,
        by_epoch=True, convert_to_iter_based=True),
]

# === 核心提速点: 开启混合精度训练 (AMP) ===
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999),
        weight_decay=weight_decay),
    clip_grad=dict(max_norm=10.0)
)

default_hooks = dict(
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=save_epoch_intervals,
        save_best='auto', max_keep_ckpts=max_keep_ckpts)
)
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, strict_load=False, priority=49),
    dict(type='mmdet.PipelineSwitchHook', switch_epoch=max_epochs - close_mosaic_epochs, switch_pipeline=train_pipeline_stage2)
]

# === 9. 评估器与训练循环 (更新评估器和总epoch) ===
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + val_ann_file, # 使用新数据集的标注文件
    metric='bbox', classwise=True,
)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs, # 使用新的总epoch数
    val_interval=save_epoch_intervals,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```
</details>

<details>
<summary>点击这里展开示例代码2</summary>

```python
# =================================================================================
# 文件名: yololow.py

_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']

# === 加载最佳模型 & 创建新工作目录 ===
# 使用 load_from 加载你找到的最好的权重，开始一次全新的优化过程
load_from ='D:/best_coco.pth'

# 为这次微调创建一个全新的、独立的工作目录
work_dir = './work_dirs/yolo'


# === 2. 数据集与类别设置 (保持不变) ===
data_root = 'D:/mmyolomain/data/'
train_ann_file = 'annotations/train.json'
train_data_prefix = 'train2017/'
val_ann_file =  'annotations/val.json'
val_data_prefix = 'val2017/'

num_classes = 3
metainfo = dict(classes=())


# === 3. 核心修改：新的微调超参数 ===
img_scale = (960, 960)

train_batch_size_per_gpu = 12
train_num_workers = 8
persistent_workers = True

# 设置一个较短的微调周期
max_epochs = 20
# 在微调阶段，可以更早或全程关闭Mosaic，这里我们设置最后10个周期关闭
close_mosaic_epochs = 10
# 每2个epoch就保存和验证一次，方便观察细微变化
save_epoch_intervals = 2


# === 4. 模型和通用参数 (保持不变) ===
dataset_type = 'YOLOv5CocoDataset'
val_batch_size_per_gpu = 1
val_num_workers = 2
batch_shapes_cfg = None
deepen_factor = 0.33
widen_factor = 0.5
strides = [8, 16, 32]
last_stage_out_channels = 1024
num_det_layers = 3
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
affine_scale = 0.4
max_aspect_ratio = 100
tal_topk = 10; tal_alpha = 0.5; tal_beta = 6.0
loss_cls_weight = 0.5; loss_bbox_weight = 7.5; loss_dfl_weight = 1.5 / 4
weight_decay = 0.05
max_keep_ckpts = 5
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


# === 5. 模型结构 (保持不变) ===
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[128., 128., 128.], std=[128., 128., 128.], bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet', arch='P5', last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor, widen_factor=widen_factor, norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(
        type='YOLOv8PAFPN', deepen_factor=deepen_factor, widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels], out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule', num_classes=num_classes, in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor, reg_max=16, norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True), featmap_strides=strides, skip_dfl=False),
        prior_generator=dict(type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='none', loss_weight=loss_cls_weight),
        loss_bbox=dict(type='IoULoss', iou_mode='ciou', bbox_format='xyxy', reduction='sum', loss_weight=loss_bbox_weight, return_iou=False),
        loss_dfl=dict(type='mmdet.DistributionFocalLoss', reduction='mean', loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner', num_classes=num_classes, use_ciou=True,
            topk=tal_topk, alpha=tal_alpha, beta=tal_beta, eps=1e-9)),
    test_cfg=dict(
        multi_label=True, nms_pre=30000, score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7), max_per_img=300))


# === 6. 数据处理流水线 (保持不变) ===
albu_train_transforms = [
    dict(type='Blur', p=0.01), dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01), dict(type='CLAHE', p=0.01)
]
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]
last_transform = [
    dict(
        type='mmdet.Albu', transforms=albu_train_transforms,
        bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={'img': 'image', 'gt_bboxes': 'bboxes'}),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction'))
]
train_pipeline = [
    *pre_transform,
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0, max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border=(-img_scale[0] // 2, -img_scale[1] // 2), border_val=(114, 114, 114)),
    *last_transform
]
train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=True, pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0, max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio, border_val=(114, 114, 114)),
    *last_transform
]


# === 7. 数据加载器 (保持不变) ===
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu, num_workers=train_num_workers,
    persistent_workers=persistent_workers, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline, metainfo=metainfo
        ))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param'))
]
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu, num_workers=val_num_workers,
    persistent_workers=persistent_workers, pin_memory=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, test_mode=True,
        data_prefix=dict(img=val_data_prefix), ann_file=val_ann_file,
        pipeline=test_pipeline, batch_shapes_cfg=batch_shapes_cfg, metainfo=metainfo
        ))
test_dataloader = val_dataloader


# === 8. 核心修改：全新的微调优化器和学习率策略 ===
# 使用一个极低的基础学习率
base_lr = 1e-5  # 0.00001

# 为微调设计的学习率调度器
param_scheduler = [
    # 在微调阶段，我们不再需要很长的预热
    dict(
        type='LinearLR',
        start_factor=1.0,  # 直接从 base_lr 开始，不打折
        by_epoch=True,
        begin=0,
        end=1),
    # 主要使用余弦退火，在新的、短的周期内缓慢下降
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.1,  # 学习率最低降到 1e-6
        begin=1,
        end=max_epochs,  # 对应新的 max_epochs (20)
        T_max=max_epochs - 1,
        by_epoch=True,
        convert_to_iter_based=True),
]

# 优化器配置保持不变，但会使用上面的新 base_lr
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999),
        weight_decay=weight_decay),
    clip_grad=dict(max_norm=10.0)
)

default_hooks = dict(
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=save_epoch_intervals, # 每2个epoch保存一次
        save_best='auto', max_keep_ckpts=max_keep_ckpts)
)
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, strict_load=False, priority=49),
    dict(type='mmdet.PipelineSwitchHook', switch_epoch=max_epochs - close_mosaic_epochs, switch_pipeline=train_pipeline_stage2)
]


# === 9. 评估器与训练循环 (适配新的周期) ===
val_evaluator = dict(
    type='mmdet.CocoMetric',
    #ann_file=data_root + val_ann_file,
    metric='bbox', classwise=True,
)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,  # 使用新的总周期数 (20)
    val_interval=save_epoch_intervals, # 使用新的验证间隔 (2)
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```
</details>

<details>
<summary>点击这里展开示例代码3</summary>

```python
# =================================================================================
# 文件名: yolo.py

_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']

# === 加载新的起点模型 & 创建新工作目录 ===
# 指定从 epoch.pth 开始，保持不变
load_from = 'D:/mmyolo-main/mmyolo/work_dirs/yolo/epoch.pth'

# 为这次【强化微调】创建一个全新的、独立的工作目录
work_dir = './work_dirs/yolo'


# === 2. 数据集与类别设置 (保持不变) ===
data_root = 'D:/mmyolo/data/'
train_ann_file = 'annotations/train.json'
train_data_prefix = 'train2017/'
val_ann_file =  'annotations/val.json'
val_data_prefix = 'val2017/'

num_classes = 2
metainfo = dict(classes=())


# === 3. 核心修改：新的强化微调超参数 ===
img_scale = (960, 960)

train_batch_size_per_gpu = 12
train_num_workers = 8
persistent_workers = True

# --- 【修改】延长训练周期，给予模型更充分的打磨时间 ---
max_epochs = 40
# --- 【修改】相应调整关闭Mosaic的时机，在最后15个周期关闭 ---
close_mosaic_epochs = 15
# 每2个epoch就保存和验证一次，方便观察细微变化
save_epoch_intervals = 2


# === 4. 核心修改：调整损失函数权重 ===
# --- 【修改】提高分类损失权重，让模型更关注难分的类别；同时略微降低BBox权重以平衡总损失 ---
loss_cls_weight = 0.8        # 从 0.5 提升
loss_bbox_weight = 7.0       # 从 7.5 降低
loss_dfl_weight = 1.5 / 4    # 保持不变

# --- 以下为保持不变的参数 ---
dataset_type = 'YOLOv5CocoDataset'
val_batch_size_per_gpu = 1; val_num_workers = 2
batch_shapes_cfg = None; deepen_factor = 0.33; widen_factor = 0.5
strides = [8, 16, 32]; last_stage_out_channels = 1024
num_det_layers = 3; norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
affine_scale = 0.4; max_aspect_ratio = 100
tal_topk = 10; tal_alpha = 0.5; tal_beta = 6.0
weight_decay = 0.05; max_keep_ckpts = 5
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


# === 5. 模型结构 (保持不变) ===
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[128., 128., 128.], std=[128., 128., 128.], bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet', arch='P5', last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor, widen_factor=widen_factor, norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(
        type='YOLOv8PAFPN', deepen_factor=deepen_factor, widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels], out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule', num_classes=num_classes, in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor, reg_max=16, norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True), featmap_strides=strides, skip_dfl=False),
        prior_generator=dict(type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='none', loss_weight=loss_cls_weight),
        loss_bbox=dict(type='IoULoss', iou_mode='ciou', bbox_format='xyxy', reduction='sum', loss_weight=loss_bbox_weight, return_iou=False),
        loss_dfl=dict(type='mmdet.DistributionFocalLoss', reduction='mean', loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner', num_classes=num_classes, use_ciou=True,
            topk=tal_topk, alpha=tal_alpha, beta=tal_beta, eps=1e-9)),
    test_cfg=dict(
        multi_label=True, nms_pre=30000, score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7), max_per_img=300))


# === 6. 核心修改：修正数据增强流水线以兼容您的环境 ===
albu_train_transforms = [
    dict(type='Blur', p=0.01), dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01), dict(type='CLAHE', p=0.01)
]
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]
last_transform = [
    dict(
        type='mmdet.Albu', transforms=albu_train_transforms,
        bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={'img': 'image', 'gt_bboxes': 'bboxes'}),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction'))
]

# --- 【重要修改】已为您修正此部分代码 ---
train_pipeline = [
    *pre_transform,
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0, pre_transform=pre_transform),
    # --- 【已修正】删除了不兼容的 'max_num_pasted' 参数，现在可以正常运行 ---
    dict(
        type='YOLOv5CopyPaste',
        prob=0.75),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0, max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border=(-img_scale[0] // 2, -img_scale[1] // 2), border_val=(114, 114, 114)),
    *last_transform
]

# 第二阶段（关闭Mosaic）的流水线保持不变
train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=True, pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0, max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio, border_val=(114, 114, 114)),
    *last_transform
]


# === 7. 数据加载器 (保持不变) ===
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu, num_workers=train_num_workers,
    persistent_workers=persistent_workers, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline, metainfo=metainfo
        ))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param'))
]
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu, num_workers=val_num_workers,
    persistent_workers=persistent_workers, pin_memory=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, test_mode=True,
        data_prefix=dict(img=val_data_prefix), ann_file=val_ann_file,
        pipeline=test_pipeline, batch_shapes_cfg=batch_shapes_cfg, metainfo=metainfo
        ))
test_dataloader = val_dataloader


# === 8. 核心修改：适配新的训练周期的优化器和学习率策略 ===
# --- 【修改】可以略微提高学习率，给模型更多动力学习新知识 ---
base_lr = 2e-5  # 0.00002

# --- 【修改】学习率调度器需适配新的 max_epochs ---
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0, by_epoch=True, begin=0, end=1),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.1,
        begin=1,
        end=max_epochs,  # 对应新的 max_epochs (40)
        T_max=max_epochs - 1,
        by_epoch=True,
        convert_to_iter_based=True),
]

# 优化器配置保持不变，但会使用上面的新 base_lr
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999),
        weight_decay=weight_decay),
    clip_grad=dict(max_norm=10.0) # 梯度裁剪依然保留，非常重要
)

default_hooks = dict(
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=save_epoch_intervals,
        save_best='auto', max_keep_ckpts=max_keep_ckpts)
)
# --- 【修改】PipelineSwitchHook 的切换时机也需要更新 ---
switch_pipeline_epoch = max_epochs - close_mosaic_epochs # 40 - 15 = 25
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, strict_load=False, priority=49),
    dict(type='mmdet.PipelineSwitchHook', switch_epoch=switch_pipeline_epoch, switch_pipeline=train_pipeline_stage2)
]


# === 9. 核心修改：评估器与训练循环 (适配新的周期) ===
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox', classwise=True,
)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,  # 使用新的总周期数 (40)
    val_interval=save_epoch_intervals,
    # --- 【修改】动态间隔的触发点也需要更新 ---
    dynamic_intervals=[(switch_pipeline_epoch, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```
</details>
