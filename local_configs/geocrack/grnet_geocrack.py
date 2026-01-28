_base_ = [
    "../_base_/datasets/GeoCrackDataset_pipeline.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_60k_adamw.py", 
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='GRNet',
        in_channels=3,
        base_channels=64,
        use_dlka=True,
        use_scse=True,
        scse_reduction=16,
        scse_min_channels=4,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        ignore_index=255,
        align_corners=False,
        loss_decode=[dict(
            type="CrossEntropyLoss",
        ), dict(
            type="DiceLoss",
        )]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)


# optimizer
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    min_lr=1e-5,
    by_epoch=False,
)

data = dict(
    samples_per_gpu=4
)
evaluation = dict(
    interval=1000,
    metric=['mIoU', 'mDice', 'mFscore'], 
    class_names = ['background', 'crack'] ,
    save_best="mIoU"
)
