scale = 2
img_size = 64
window_size = 8
height = 520
width = 368
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='swinIR',
        upscale=2,
        img_size=(64, 64),
        window_size=8,
        img_range=1.0,
        depths=[2, 2, 2, 2],
        embed_dim=32,
        num_heads=[2, 2, 2, 2],
        mlp_ratio=2,
        upsampler='pixelshuffle'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=2)
train_dataset_type = 'SRFolderDataset'
val_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=False),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=False),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='SRFolderDataset',
        lq_folder='../datasets/train/LRx2',
        gt_folder='../datasets/train/HRx2',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='lq',
                flag='unchanged'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='gt',
                flag='unchanged'),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(
                type='Normalize',
                keys=['lq', 'gt'],
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=False),
            dict(type='PairedRandomCrop', gt_patch_size=128),
            dict(
                type='Flip',
                keys=['lq', 'gt'],
                flip_ratio=0.5,
                direction='horizontal'),
            dict(
                type='Flip',
                keys=['lq', 'gt'],
                flip_ratio=0.5,
                direction='vertical'),
            dict(
                type='RandomTransposeHW',
                keys=['lq', 'gt'],
                transpose_ratio=0.5),
            dict(
                type='Collect',
                keys=['lq', 'gt'],
                meta_keys=['lq_path', 'gt_path']),
            dict(type='ImageToTensor', keys=['lq', 'gt'])
        ],
        scale=2,
        filename_tmpl='{}'),
    val=dict(
        type='SRFolderDataset',
        lq_folder='../datasets/test/LRx2',
        gt_folder='../datasets/test/HRx2',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='lq',
                flag='unchanged'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='gt',
                flag='unchanged'),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(
                type='Normalize',
                keys=['lq', 'gt'],
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=False),
            dict(
                type='Collect',
                keys=['lq', 'gt'],
                meta_keys=['lq_path', 'gt_path']),
            dict(type='ImageToTensor', keys=['lq', 'gt'])
        ],
        scale=2,
        filename_tmpl='{}'),
    test=dict(
        type='SRFolderDataset',
        lq_folder=' ../datasets/test/LRx2',
        gt_folder='../datasets/test/HRx2',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='lq',
                flag='unchanged'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='gt',
                flag='unchanged'),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(
                type='Normalize',
                keys=['lq', 'gt'],
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=False),
            dict(
                type='Collect',
                keys=['lq', 'gt'],
                meta_keys=['lq_path', 'gt_path']),
            dict(type='ImageToTensor', keys=['lq', 'gt'])
        ],
        scale=2,
        filename_tmpl='{}'))
optimizers = dict(generator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
lr_config = dict(policy='poly', power=0.9, min_lr=5e-05, by_epoch=False)
evaluation = dict(interval=1000)
checkpoint_config = dict(interval=10000, save_optimizer=False, by_epoch=False)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None
total_iters = 20000
cudnn_benchmark = False
find_unused_parameters = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs_2x/swinir2x'
gpus = 1
