crop_size = (
    512,
    512,
)
norm_cfg = dict(requires_grad=True, type='SyncBN')
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
# data_root = 'F:/cosas/COSAS24-TrainingSet/task1'
# dataset_type = 'COSASDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=16000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=32,
        in_channels=3,
        init_cfg=None,
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            2,
            2,
            2,
            2,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            32,
            64,
            160,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=norm_cfg,
        num_classes=150,
        type='SegformerHead'),
    pretrained=None,
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
    train_cfg=dict(),
    type='EncoderDecoder')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=160000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(cat_max_ratio=0.75, crop_size=crop_size, type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

test_pipeline = val_pipeline

train_cfg = dict(max_iters=160000, type='IterBasedTrainLoop', val_interval=16000)
# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     dataset=dict(
#         type='ConcatDataset',
#         datasets=[dict(
#             ann_file=f'{domain}_train.txt',
#             data_prefix=dict(img_path=f'{domain}/image', seg_map_path=f'{domain}/mask'),
#             data_root=data_root,
#             pipeline=train_pipeline,
#             type=dataset_type
#         ) for domain in ['colorectum', 'stomach']]
#     ),
#     persistent_workers=True,
#     sampler=dict(shuffle=True, type='InfiniteSampler')
# )

val_cfg = dict(type='ValLoop')
# val_dataloader = dict(
#     batch_size=1,
#     dataset=dict(
#         type='ConcatDataset',
#         datasets=[dict(
#             ann_file=f'{domain}_validator.txt',
#             data_prefix=dict(img_path=f'{domain}/image', seg_map_path=f'{domain}/mask'),
#             data_root=data_root,
#             pipeline=val_pipeline,
#             type=dataset_type
#         ) for domain in ['colorectum', 'stomach', 'pancreas']]
#     ),
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(shuffle=False, type='DefaultSampler')
# )

tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]


val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]

resume = False
test_cfg = dict(type='TestLoop')
# test_dataloader = val_dataloader
test_evaluator = val_evaluator


visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
])
