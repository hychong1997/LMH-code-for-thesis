_base_ = ['./roi_trans_swin_tiny_fpn_1x_VSAI_rr_le90.py']

data_root = '/home/wk/Data/VSAI/VSAI1320/ms/'

data = dict(
    train=dict(
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/'),
    val=dict(
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/'),
    test=dict(
        ann_file='/home/wk/Data/UAV_ROD_Data/test/test/images/',
        img_prefix='/home/wk/Data/UAV_ROD_Data/test/test/images/'))