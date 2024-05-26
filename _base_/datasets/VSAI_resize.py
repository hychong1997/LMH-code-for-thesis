# dataset settings
dataset_type = 'VSAIDataset_resize'
data_root = '/home/wk/Data/VSAI/VSAI1320/ss/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
angle_version = 'le90'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),# keep_ratio=True),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),  #
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9, 11],
        version=angle_version),  #
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='RandomOBBRotate', rotate_after_flip=True, angles=(0, 0), vert_rate=1.),
    dict(type='Pad', size_divisor=32),
    #dict(type='Mask2OBB', obb_type='obb'),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'height', 'pitch', 'exposure'])
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',#type='MultiScaleFlipRotateAug',
        img_scale=[(800, 800)],
        flip=False,
        # rotate=False,
        transforms=[
            # dict(type='RResize', keep_ratio=True),
            dict(type='RResize', img_scale=(800, 800)),
            # dict(type='OBBRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='RandomOBBRotate', rotate_after_flip=True),
            dict(type='Pad', size_divisor=32),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/',
        # attribute= data_root + 'trainval/attributes/',
        img_prefix=data_root + 'trainval/images (copy)/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/',
        #attribute= data_root + 'test/attributes/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/',
        #attribute= data_root + 'test/attributes/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))