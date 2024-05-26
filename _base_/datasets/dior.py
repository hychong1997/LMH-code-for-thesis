dataset_type = 'DIORDataset'
data_root = 'data/DIOR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),# keep_ratio=True),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='RandomOBBRotate', rotate_after_flip=True, angles=(0, 0), vert_rate=1.),
    dict(type='Pad', size_divisor=32),
    #dict(type='Mask2OBB', obb_type='obb'),
    dict(type='DefaultFormatBundle'),
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
            dict(type='RResize'),
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
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/Annotations/OBBtxt/trainval/',
        img_prefix=data_root + 'JPEGImages-trainval/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/Annotations/OBBtxt/test/',
        img_prefix=data_root + 'JPEGImages-test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/Annotations/OBBtxt/test/',
        img_prefix=data_root + 'JPEGImages-test/',
        pipeline=test_pipeline))


# does evaluation while training
# uncomments it  when you need evaluate every epoch
# data = dict(
    # samples_per_gpu=2,
    # workers_per_gpu=4,
    # train=dict(
        # type=dataset_type,
        # xmltype='obb',
        # imgset=data_root + 'ImageSets/Main/trainval.txt',
        # ann_file=data_root + 'Annotations/',
        # img_prefix=data_root + 'JPEGImages/',
        # pipeline=train_pipeline),
    # val=dict(
        # type=dataset_type,
        # xmltype='obb',
        # imgset=data_root + 'ImageSets/Main/test.txt',
        # ann_file=data_root + 'Annotations/',
        # img_prefix=data_root + 'JPEGImages/',
        # pipeline=test_pipeline),
    # test=dict(
        # type=dataset_type,
        # xmltype='obb',
        # imgset=data_root + 'ImageSets/Main/test.txt',
        # ann_file=data_root + 'Annotations/',
        # img_prefix=data_root + 'JPEGImages/',
        # pipeline=test_pipeline))
# evaluation = dict(metric='mAP')

# disable evluation, only need train and test
# uncomments it when use trainval as train
"""
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        xmltype='obb',
        imgset=data_root + 'ImageSets/Main/trainval.txt',
        ann_file=data_root + 'Annotations/obb',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        xmltype='obb',
        imgset=data_root + 'ImageSets/Main/test.txt',
        ann_file=data_root + 'Annotations/obb',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=test_pipeline))
evaluation = None
"""