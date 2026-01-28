# dataset settings
dataset_type = "GeoCrackDataset"  # 数据集类名
data_root = (
    "data/GeoCrack/"  # 数据集根目录（相对于mmsegementation主目录）
)

img_norm_cfg = dict(
    mean=[142.822, 140.594, 133.469],
    std=[48.271, 48.646, 49.566],
    to_rgb=True
)

crop_size = (224, 224)

# 数据预处理管道
# 包含图像加载、标注加载、图像缩放、随机裁剪、随机翻转、光度畸变、归一化、填充格式化输出和数据收集等步骤
# 这些步骤是为了将原始图像和标注转换为模型可以接受的格式，并进行必要的数据增强
# 注意：如果使用了`RandomCrop`，则需要确保图像和标注的尺寸一致，且裁剪后的图像和标注大小相同
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="Resize", img_scale=(224, 224), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5, direction='horizontal'),
    dict(type="RandomRotate", prob=0.5, degree=(-15, 15)),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

# 测试/验证数据的预处理流程
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type="Resize",img_scale=crop_size, keep_ratio=True),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# DataLoader配置
# 定义训练、验证和测试数据集的加载方式，包括每个GPU的样本数、工作线程数等
# 这里的`samples_per_gpu`表示每个GPU加载的样本数量，`workers_per_gpu`表示每个GPU使用的工作线程数
# `train`、`val`和`test`分别定义了训练集、验证集和测试集的数据加载方式
# `type`指定数据集的类型，`
data = dict(
    samples_per_gpu=4,# 单个 GPU 的 Batch size
    workers_per_gpu=4,# 单个 GPU 分配的数据加载线程数
    train=dict(
        type=dataset_type,# 数据集的类别
        data_root=data_root,# 数据集的根目录
        img_dir="img_dir/train",# 数据集图像的文件夹
        ann_dir="ann_dir/train",# 数据集注释的文件夹
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="img_dir/val",
        ann_dir="ann_dir/val",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="img_dir/test",
        ann_dir="ann_dir/test",
        pipeline=test_pipeline,
    ),
)
