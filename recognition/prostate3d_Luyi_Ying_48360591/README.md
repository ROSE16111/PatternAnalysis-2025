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
- `tools_count_labels.py` – count the number of each class
- `quick_check.py` – read a NIfTI，check data and lables
- `README.md`   – this document
- `pics` picture resources for readme file and output from other files

## Environment
- Python 3.10, PyTorch (CUDA 11.8), torch: 2.2.2; nibabel, numpy, scikit-image, torchio, matplotlib.
```bash
# example - 3D Medical Augmentation Library
pip install nibabel numpy scikit-image matplotlib torchio
# or: pip install monai
```
## device:
* NVIDIA-SMI 527.41 
* Driver Version: 527.41 
* CUDA Version: 12.0 2G

note: use patch for run in GPU with low storage
## data
`semantic_MRs_anon/` -  3D MRI volumetric images, X

`semantic_labels_anon/` - 3D semantic tags, Y

* **canonical id**
    
    e.g:Case_013_Week3_LFOV
* **Image shape**(C,Z,Y,X):
  * C=1（Single-channel）, channel-first 
  * voxel dimension(Z=256, Y=256, X=128)
* **Label uniques**: 
  * 0：background (60.35%)
  * 1: body （35.47%）
  * 2: bone（3.37%）
  * 3: bladder（0.57%）
  * 4: rectum （0.14%）
  * 5: prostate（0.10%）
* num_classes=6
## result:
Results on Test Set
## Testing Instructions
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

**imporved 1**:

In each epoch, only one random patch is taken from each sample. In the next epoch, another patch is taken from a different location within the same volume (randomness + data augmentation). Each epoch uses a center-patch quick check, and after N epochs, a full-scale sliding window verification is performed, saving best.ckpt with the "full-scale metric" as the standard.

  * Do light weight training
  * Replace the loss with CE + Dice (weight 0.5 is relatively stable).

**imporved 2**:
  * change random sampling to balanced sampling to avoid sampling bias
  * set mdice_full_org(get rid of background) as the best indicator

**improved 3**
  * Add category weights to CE
  * Dice is excluded from the background, and small organs are given greater weight.
  * Corrected "Union sampling" → "Class-based equalization sampling"

**visualize**:
* training_losses.png
  * Total loss = CE + 0.5*DiceLoss
* val_mdice.png
  * Val mDice (all) ： Average Dice with background (6 categories)
  * Val mDice (organs) ：The background was removed and only the average of the 5 organs was taken (which is more representative of organ performance).
* val_per_class_dice.png : The six lines represent the Dice of background/body/bone/bladder/rectum/prostate.


### `predict.py`
Inference/Derived Prediction
* load weight
* Use `split=test` to perform forward processing on each sample; `argmax` to obtain the semantic labels.
* Save the original image as an NIfTI (space-aligned) file using affine at `runs/preds/`.

### process
1. Data reading & ID matching
2. Training pipeline (patch-based)
3. Sliding window inference (GPU friendly) and successful NIfTI export.