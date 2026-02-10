# model settings
scale = 4
img_size = 64
window_size = 8
height = (1024 // scale // window_size + 1) * window_size
width = (720 // scale // window_size + 1) * window_size
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='swinIR',
        upscale=scale,
        img_size=(64, 64),
        window_size=window_size,
        img_range=1.,
        depths=[2, 2, 2, 2],
        embed_dim=16,
        num_heads=[2, 2, 2, 2],
        mlp_ratio=2,
        upsampler='pixelshuffle'
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
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
    dict(type='PairedRandomCrop', gt_patch_size=img_size*scale),
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
        type=train_dataset_type,
        lq_folder='../datasets/train/LRx4',
        gt_folder='../datasets/train/HRx4',
        pipeline=train_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    val=dict(
        type=val_dataset_type,
        lq_folder='../datasets/test/LRx4',
        gt_folder='../datasets/test/HRx4',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folder='../datasets/test/LRx4',
        gt_folder='../datasets/test/HRx4',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,  # The power of polynomial decay.
    min_lr=5e-5,  # The minimum learning rate to stable the training.
    by_epoch=False,  # Whethe count by epoch or not.)
)
evaluation = dict(interval=1000)
# checkpoint saving
checkpoint_config = dict(interval=10000, save_optimizer=False, by_epoch=False)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None
# runtime settings
total_iters = 20000
cudnn_benchmark = False
find_unused_parameters = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = f'./work_dirs_4x/{{ fileBasenameNoExtension }}'
