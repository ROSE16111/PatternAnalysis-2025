# Prostate 3D Segmentation (Normal → Hard)

## Overview
Segment (downsampled) Prostate 3D dataset with a 3D UNet baseline (Normal) aiming for per-class Dice ≥ 0.70 on test set; optionally upgrade to Improved UNet3D (Hard).

* Dice similarity coefficient
Measuring the degree of overlap between predictions and true values;The closer to 1, the more overlap.

    ![alt text](pics/image.png)
## file structures
- `modules.py`  – model components (3D UNet baseline)
- `dataset.py`  – NIfTI I/O, read and preprocessing data
- `train.py`    – training/validation loop; logs loss & per-class Dice; saves curves
- `predict.py`  – sliding-window inference; saves NIfTI + slice PNGs
- `README.md`   – this document
- `pics` picture resources for readme file

## Environment
- Python 3.10, PyTorch (CUDA 11.8), nibabel, numpy, scikit-image, torchio/monai (one of them), matplotlib.
```bash
# example - 3D Medical Augmentation Library
pip install nibabel numpy scikit-image matplotlib torchio
# or: pip install monai
```

## data
`semantic_MRs_anon/` -  3D MRI volumetric images, X

`semantic_labels_anon/` - 3D semantic tags, Y

* **canonical id**
    
    e.g:Case_013_Week3_LFOV
* **Image shape**(C,Z,Y,X):
  * C=1（Single-channel）, channel-first 
  * voxel dimension(Z=256, Y=256, X=128)
* **Label uniques**: 
  * 0: body
  * 1: bone
  * 2: bladder
  * 3: rectum
  * 4: prostate
* num_classes=5

## code:
### `dataset.py`
Find the correct file, match the image with the label, normalize the ID, split the data, read the NIfTI, perform normalization, and return the standard tensor.
return  {"id", "image", "label", "affine"}
* import
  * `nibabel`：Reading and writing NIfTI (.nii/.nii.gz) medical 3D volume data
  * `torch.utils.data.Dataset`: Custom PyTorch Dataset Base Class
* `load_nii(path)`
  * Open NIfTI, and use get_fdata() to read it as a three-dimensional array (Z, Y, X) of float32.
  * Returns (data, affine) - (3D , voxel→world)
* `class Prostate3DDataset(Dataset)`
  * Automatic directory adaptation
  * Scan two directories and canonicalize the filenames.
    * `self.img_index = {canonical_id: Path}`
    * `self.lab_index = {canonical_id: Path}`
  * `splits.json`: With a fixed random seed of 42, common_ids 80/10/10 are divided into train/val/test to ensure experimental reproducibility.
### `modules.py`
A minimal runnable 3D U-Net（encoder*3 –bottleneck–decoder *4 + skip connections. Use InstanceNorm3d + LeakyReLU
* **Output** the logits for num_classes channels.
* The parameter `base` can be adjusted to adapt to the video memory.
### `train.py`
training+validation
* Construct a DataLoader using Prostate3DDataset(split=train/val)
* use
  * CrossEntropyLoss baseline; 
  * AdamW optimization; 
  * can enable --amp mixed precision.
* Calculate per-class Dice for each epoch, print the table, and save best.ckpt - mean Dice(val) at `runs/best.ckpt`
### `predict.py`
Inference/Derived Prediction
* load weight
* Use `split=test` to perform forward processing on each sample; `argmax` to obtain the semantic labels.
* Save the original image as an NIfTI (space-aligned) file using affine at `runs/preds/`.