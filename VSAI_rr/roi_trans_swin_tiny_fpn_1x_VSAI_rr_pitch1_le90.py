_base_ = ['./roi_trans_swin_tiny_fpn_1x_VSAI_rr_le90.py']

data_root = '/home/wk/Data/VSAI/VSAI1320/'

data = dict(
    train=dict(
        ann_file=data_root + 'ss/trainval/annfiles/',
        img_prefix=data_root + 'ss/trainval/images/'),
    val=dict(
        ann_file=data_root + 'ss/test/annfiles/',
        img_prefix=data_root + 'ss/test/images/'),
    test=dict(
        ann_file=[data_root + 'pitch/test/-30_0/annfiles/', data_root + 'pitch/test/-60_-30/annfiles/',data_root + 'pitch/test/-90_-60/annfiles/'],
        img_prefix=[data_root + 'pitch/test/-30_0/images/', data_root + 'pitch/test/-60_-30/images/',data_root + 'pitch/test/-90_-60/images/'])
)
