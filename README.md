# YOLO-HDS Reproducibility Materials

This repository provides the supplementary annotations, exact dataset splits, model configuration, custom module implementations, training settings, and near-duplicate screening code used in the revised YOLO-HDS study.

## Contents

```text
code/                       HWD, DySample, and SIoU implementations
configs/yolo-hds.yaml       YOLO-HDS model configuration
configs/train.py            Training entry point and principal settings
configs/melon.yaml          Dataset configuration and class names
splits/train.txt            Exact 3,211-image training split
splits/val.txt              Exact 400-image validation split
splits/test.txt             Final 389-image test split
supplementary_labels/       205 manually added YOLO-format annotations
screening/                  Near-duplicate screening script
requirements.txt            Python dependencies
```

The released materials correspond to the reviewer's reproducibility requests as follows:

| Requested material | Repository location |
|---|---|
| Supplementary annotations for previously unlabeled images | `supplementary_labels/` |
| Exact data splits | `splits/train.txt`, `splits/val.txt`, `splits/test.txt` |
| Proposed-module code | `code/AddBlocks.py`, `code/siou.py` |
| Model configuration | `configs/yolo-hds.yaml` |
| Training settings | `configs/train.py` |
| Near-duplicate screening method | `screening/near_duplicate_screening.py` |

## Dataset and annotations

The images and original annotations are from the public [Melon Ripeness Detection](https://github.com/XuebinJing/Melon-Ripeness-Detection) dataset. After checking and reorganizing the downloaded files, 4,011 usable images were obtained.

The images were randomly divided at the image level using an 8:1:1 ratio. The split was not class-stratified. The original partition contained 3,211 training images, 400 validation images, and 400 test images. Following near-duplicate screening, 11 affected test images were excluded, leaving the final 389-image test split supplied here. All results reported for the revised test set use these 389 images.

The class mapping is:

```text
0: unripe
1: ripe
```

The original annotations supplied with the public dataset were retained. Images without corresponding annotation files were manually annotated using the same two classes and YOLO normalized bounding-box format:

```text
class_id x_center y_center width height
```

The 205 added annotation files are provided in `supplementary_labels/`.  Of these files, 201 correspond to images in the final splits, while four correspond to images excluded during duplicate screening. Their class identifiers and normalized coordinates were checked before release.

Images are not redistributed. Download the original dataset, use `splits/*.txt` to reconstruct the three subsets, and place the corresponding images and labels under:

```text
train/images     train/labels
val/images       val/labels
test/images      test/labels
```

## Model code

YOLO-HDS is based on YOLO11n and uses:

- Haar wavelet downsampling (HWD) in the backbone;
- DySample in LP mode in the detection head;
- SIoU for bounding-box regression.

`code/AddBlocks.py` contains the HWD and DySample implementations, while `code/siou.py` contains the SIoU implementation. The complete YOLO-HDS architecture is specified in `configs/yolo-hds.yaml`. 

## Environment

The reference training environment was:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| CUDA used by PyTorch | 12.8 |
| cuDNN | 9.10.2 |
| Ultralytics | 8.4.2 |

Install the base framework and remaining packages with:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ultralytics==8.4.2
python -m pip install -r requirements.txt
```

## Training and augmentation settings

The principal training settings were:

| Setting | Value |
|---|---:|
| Input size | 640 × 640 |
| Epochs | 100 |
| Batch size | 64 |
| Optimizer | AdamW |
| Pretrained weights | False |
| AMP | True |
| Seed | 0 |
| Close Mosaic | Final 10 epochs |

Unless otherwise specified, the remaining training and data-augmentation settings followed the default configuration of Ultralytics v8.4.2.

## Near-duplicate screening

`screening/near_duplicate_screening.py` computes SHA-256 hashes for exact-duplicate checking and 63-bit perceptual hashes based on the low-frequency DCT coefficients of 32 × 32 grayscale images. Training-test and validation-test pairs with a pHash Hamming distance of 10 or less were treated as candidates and manually inspected. Eleven affected test images were excluded from final evaluation. The script produces `near_duplicate_pairs.csv`. 

The supplied files document the exact final data split, added annotations, model structure, custom implementations, software environment, and training protocol. Small numerical differences may occur across GPU, driver, CUDA, cuDNN, and PyTorch versions.
