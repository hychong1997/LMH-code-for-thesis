_base_ = ['./roi_trans_swin_tiny_fpn_1x_VSAI_rr_le90.py']

data_root1 = '/home/wk/Data/VSAI/VSAI1320/ss/'
data_root2 = '/home/wk/Data/VSAI/VSAI1320/mosaic/merge/'


data = dict(
    train=dict(
        ann_file=[data_root1 + 'trainval/annfiles/', data_root2 + 'val/annfiles/', data_root2 + 'train/annfiles/'],
        img_prefix=[data_root1 + 'trainval/images/', data_root2 + 'val/images/', data_root2 + 'train/images/']),
    val=dict(
        ann_file=data_root1 + 'test/annfiles/',
        img_prefix=data_root1 + 'test/images/'),
    test=dict(
        ann_file=data_root1 + 'test/annfiles/',
        img_prefix=data_root1 + 'test/images/'))