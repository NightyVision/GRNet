_base_ = [
    "../../_base_/datasets/RockCrackFold0Dataset_pipeline.py",  # 数据集配置（ADE20K，带重复采样）
    "../../_base_/default_runtime.py",  # 运行时配置（日志、钩子等
    "../../_base_/schedules/schedule_2k_adamw.py",  # 优化器与学习率调度配置
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
        use_dlka=False,
        frozen_stages=-1,
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
        norm_eval=False,
    ),
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
    _delete_=True,  # 删除基础配置中的optimizer
    type="AdamW",  # 优化器类型
    lr=0.00006,  # 学习率
    betas=(0.9, 0.999),  # AdamW的beta参数
    weight_decay=0.01,  # 权重衰减
    paramwise_cfg=dict(  # 参数分组设置
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=5.0),
        }
    ),
)

lr_config = dict(
    _delete_=True,  # 删除基础配置中的lr_config
    policy="poly",  # 多项式学习率策略
    warmup="linear",  # 线性预热
    warmup_iters=50,  # 预热步数
    warmup_ratio=1e-6,  # 预热起始比例
    power=1.0,  # poly策略的幂
    min_lr=0.0,  # 最小学习率
    by_epoch=False,  # 按iteration调整学习率
)

data = dict(
    samples_per_gpu=4
)  # total batch size 16 每张GPU的batch size为4（总batch size=4*GPU数）

evaluation = dict(
    interval=40,  # 每8000次迭代评估一次
    metric=['mIoU', 'mDice', 'mFscore'], 
    class_names = ['background', 'crack'] ,
    save_best="mIoU"
)

log_config = dict(
    interval=10,  # 每 10 次迭代打印一次（约半个 Epoch）
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # 建议打开 Tensorboard，看 Loss 曲线更直观
        # dict(type='TensorboardLoggerHook') 
    ]
)