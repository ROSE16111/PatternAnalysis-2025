# -*- coding: utf-8 -*-
r"""
Predict on test set (imagesTs) and save NIfTI predictions.
默认整幅推理（内存足够时）。若报显存不足，可改为滑窗（sliding window）。

# 2 个 epoch 试跑，确认能完整走通
python recognition\prostate3d_Luyi_Ying_48360591\predict.py --data_root "D:\document\UQ\4COMP3710\A3\data" --ckpt runs\best.ckpt --outdir runs\preds

"""
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch

from dataset import Prostate3DDataset
from modules import UNet3D

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 载入模型
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"]
    num_classes = state["outc.weight"].shape[0] 
    base = ckpt.get("args", {}).get("base", 16)
    model = UNet3D(in_ch=1, num_classes=num_classes, base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # dataset（test）
    ds = Prostate3DDataset(args.data_root, split="test")

    for i in range(len(ds)):
        sample = ds[i]
        img = sample["image"].unsqueeze(0).to(device)  # (1,1,Z,Y,X)
        affine = sample["affine"]
        pid = sample["id"]

        with torch.no_grad():
            logits = model(img)              # (1,C,Z,Y,X)
            pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Save the image at the same size as the original. NIfTI
        out_path = outdir / f"{pid}_pred.nii.gz"
        nib.save(nib.Nifti1Image(pred, affine=affine), str(out_path))
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="runs/best.ckpt")
    ap.add_argument("--outdir", type=str, default="runs/preds")
    ap.add_argument("--base", type=int, default=16)
    args = ap.parse_args()
    main(args)
