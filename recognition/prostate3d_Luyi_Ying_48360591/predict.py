r"""
python recognition\prostate3d_Luyi_Ying_48360591\predict.py --data_root "D:\document\UQ\4COMP3710\A3\data" --ckpt runs\best.ckpt --outdir runs\preds --device cuda --patch 64 64 64 --overlap 16 --amp

"""
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from dataset import Prostate3DDataset
from modules import UNet3D

def compute_starts(L, P, O):
    if L <= P:
        return [0]
    stride = max(P - O, 1)
    starts = list(range(0, L - P + 1, stride))
    if starts[-1] != L - P:
        starts.append(L - P)
    return starts

@torch.no_grad()
def sliding_window_predict(model, img, num_classes, patch, overlap, device, amp=True):
    """
    img: (1,1,Z,Y,X) torch.float32 on 'device'
    return: (Z,Y,X) uint8
    """
    _, _, Z, Y, X = img.shape
    pz, py, px = patch
    sz_list = compute_starts(Z, pz, overlap)
    sy_list = compute_starts(Y, py, overlap)
    sx_list = compute_starts(X, px, overlap)

    # 累加概率，再平均
    probs_sum = np.zeros((num_classes, Z, Y, X), dtype=np.float32)
    count_map = np.zeros((Z, Y, X), dtype=np.float32)

    model.eval()
    autocast = torch.cuda.amp.autocast if (device.type == "cuda") else torch.cpu.amp.autocast
    for z0 in sz_list:
        for y0 in sy_list:
            for x0 in sx_list:
                z1, y1, x1 = z0 + pz, y0 + py, x0 + px
                tile = img[:, :, z0:z1, y0:y1, x0:x1]  # (1,1,pz,py,px)
                with autocast(enabled=amp):
                    logits = model(tile)                # (1,C,*,*,*)
                    tile_probs = F.softmax(logits, dim=1)[0].float().cpu().numpy()  # (C,*,*,*)
                cz, cy, cx = tile_probs.shape[1:]
                probs_sum[:, z0:z0+cz, y0:y0+cy, x0:x0+cx] += tile_probs
                count_map[z0:z0+cz, y0:y0+cy, x0:x0+cx] += 1.0

    probs = probs_sum / np.maximum(count_map[None, ...], 1e-6)
    pred = probs.argmax(0).astype(np.uint8)
    return pred

def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 加载权重 & 建模（从权重里推断 num_classes 与 base）
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"]
    num_classes = state["outc.weight"].shape[0]
    base = ckpt.get("args", {}).get("base", 16)

    model = UNet3D(in_ch=1, num_classes=num_classes, base=base).to(device)
    model.load_state_dict(state)
    model.eval()

    # 数据（test split）
    ds = Prostate3DDataset(args.data_root, split="test")

    patch = tuple(args.patch)
    print(f"[INFO] Device={device}, patch={patch}, overlap={args.overlap}, num_classes={num_classes}")

    for i in range(len(ds)):
        s = ds[i]
        img = s["image"].unsqueeze(0).to(device)  # (1,1,Z,Y,X)
        affine = s["affine"]; pid = s["id"]

        # 滑窗推理（AMP 在 cuda 上启用）
        pred = sliding_window_predict(
            model, img, num_classes=num_classes,
            patch=patch, overlap=args.overlap, device=device, amp=args.amp
        )

        out_path = Path(args.outdir) / f"{pid}_pred.nii.gz"
        nib.save(nib.Nifti1Image(pred, affine=affine), str(out_path))
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="runs/best.ckpt")
    ap.add_argument("--outdir", type=str, default="runs/preds")
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--patch", type=int, nargs=3, default=[64,64,64])
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()
    main(args)
