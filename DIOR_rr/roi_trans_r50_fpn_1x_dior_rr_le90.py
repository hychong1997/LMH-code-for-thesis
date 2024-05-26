_base_ = ['./roi_trans_r50_fpn_1x_dior_le90.py']

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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    train=dict(
        pipeline=train_pipeline))

