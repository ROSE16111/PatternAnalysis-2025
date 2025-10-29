# Prostate 3D Segmentation (Normal → Hard)

## Overview
Segment (downsampled) Prostate 3D dataset with a 3D UNet baseline (Normal) aiming for per-class Dice ≥ 0.70 on test set; optionally upgrade to Improved UNet3D (Hard).

* Dice similarity coefficient
Measuring the degree of overlap between predictions and true values;The closer to 1, the more overlap.

    ![alt text](pics/image.png)
## file structures
- `modules.py`  – model components (3D UNet baseline)
- `dataset.py`  – NIfTI I/O, preprocessing, will create List of training/validation/testing components(split.js)
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