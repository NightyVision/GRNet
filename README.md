# GRNet: Cross-Domain Few-Shot Segmentation of Geological Outcrop Cracks


## Abstract

Geological outcrop crack segmentation from UAV imagery is challenging due to the thin, low-contrast, and highly branched nature of cracks, as well as limited annotations and strong cross-domain variations. We propose **GRNet**, a U-shaped semantic segmentation network for geological outcrop crack extraction under **cross-domain few-shot settings**. GRNet introduces a **Geological Large-kernel Feature Attention (GLFA)** module to enhance long-range structural representation and integrates **SCSE** modules to improve crack saliency under complex backgrounds. A source-domain pretraining and target-domain fine-tuning strategy is further adopted. Experiments on the **GeoCrack** and **RockCrack** datasets show that GRNet consistently outperforms representative CNN-, Transformer-, and hybrid baselines, especially in detecting thin and weak crack segments.

## Data Preparation
We train and evaluate GRNet on two datasets: **GeoCrack** and **RockCrack**.

- **GeoCrack** is a large-scale, publicly available dataset for geological outcrop crack segmentation and is used as the **source domain** for pretraining.
- **RockCrack** is a label-scarce dataset constructed in this study from UAV photogrammetric imagery and is used as the **target domain** for few-shot fine-tuning and evaluation.

Please organize the dataset directory as follows:

```shell
data
├── GeoCrack
│   ├── img_dir
│   │   ├── train
│   │   ├── val
│   │   └── test
│   ├── ann_dir
│       ├── train
│       ├── val
│       └── test
└── RockCrack
    ├── Fold0
    │   ├── img_dir
    │	│	├── train
    │	│	├── val
    │   └── ann_dir
    ├── Fold1
    ├── Fold2
    ├── Fold3
    └── Fold4
```

## Usage

We use the [MMSegmentation]() framework to build our model. Please refer to the instructions of MMSegmentation to install the runtime environment.

To train the model, run:

```sh
python tools/train.py local_configs/geocrack/grnet_geocrack.py --work-dir outputs/GeoCrack_GRNet_6w
```

All training logs will be saved in `./work_dirs` folder by default.
