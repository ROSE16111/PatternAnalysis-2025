r"""
predict.py — 3D UNet inference & visualization (supports datasets with or without GT)

Outputs per run:
  1) <outdir>/{pid}_pred.nii.gz                   # 3D predicted label volume
  2) <outdir>/predictions_side_by_side.png        # Side-by-side: MRI / (GT) / Pred
  3) <outdir>/predictions_overlay.png             # MRI blended with colored segmentation (+ red contour for prostate)

Notes:
- If the chosen split has no ground-truth labels (e.g., test), Dice is skipped and
  Figure 1 becomes two columns (MRI / Pred).
- If the split has GT (e.g., val), Dice per class is printed and Figure 1 uses three columns.

Examples (PowerShell):
# With GT (validation split): three columns + Dice metrics
python recognition\prostate3d_Luyi_Ying_48360591\predict.py `
  --data_root "D:\document\UQ\4COMP3710\A3\data" `
  --split val `
  --ckpt runs\best.ckpt --outdir runs\preds_val --device cuda `
  --patch 64 64 64 --overlap 16 --amp --num_samples 4 --axis z --slice center

# Likely without GT (test split): two columns (MRI / Pred), overlay still available
python recognition\prostate3d_Luyi_Ying_48360591\predict.py `
  --data_root "D:\document\UQ\4COMP3710\A3\data" `
  --split test `
  --ckpt runs\best.ckpt --outdir runs\preds_test --device cuda `
  --patch 64 64 64 --overlap 16 --amp --num_samples 4 --axis z --slice center
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # Headless save
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dataset import Prostate3DDataset
from modules import UNet3D
import csv


# ----- Constants -----
NUM_CLASSES = 6
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
PROSTATE_IDX = 5  # index of prostate class in your label mapping


# ----- Sliding-window helpers (aligned with train.py) -----
def _starts(L, P, O):
    """Start indices for sliding window along a single axis."""
    if L <= P:
        return [0]
    stride = max(P - O, 1)
    s = list(range(0, L - P + 1, stride))
    if s[-1] != L - P:
        s.append(L - P)
    return s

def _blend_weight(pz, py, px):
    """3D Hanning window to smoothly blend patch probabilities."""
    wz = np.hanning(pz)[:, None, None]
    wy = np.hanning(py)[None, :, None]
    wx = np.hanning(px)[None, None, :]
    w = (wz * wy * wx).astype(np.float32)
    w /= (w.max() + 1e-8)
    return w  # shape: (pz, py, px)

@torch.no_grad()
def sliding_window_predict_probs(model, img, num_classes, patch, overlap, device, amp=True):
    """
    Run sliding-window inference and return per-class probabilities.
    Args:
        model: 3D UNet
        img:   torch.FloatTensor (1,1,Z,Y,X) on device
    Returns:
        probs: np.float32 array (C,Z,Y,X) in [0,1]
    """
    _, _, Z, Y, X = img.shape
    pz, py, px = patch
    sz, sy, sx = _starts(Z, pz, overlap), _starts(Y, py, overlap), _starts(X, px, overlap)

    probs_sum = np.zeros((num_classes, Z, Y, X), dtype=np.float32)
    count_map = np.zeros((Z, Y, X), dtype=np.float32)
    w_patch = _blend_weight(pz, py, px)

    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    model.eval()
    for z0 in sz:
        for y0 in sy:
            for x0 in sx:
                tile = img[:, :, z0:z0+pz, y0:y0+py, x0:x0+px]  # (1,1,pz,py,px)
                with autocast(enabled=amp):
                    logits = model(tile)                         # (1,C,*,*,*)
                    probs_tile = F.softmax(logits, dim=1)[0].float().cpu().numpy()  # (C,*,*,*)
                cz, cy, cx = probs_tile.shape[1:]
                w = w_patch[:cz, :cy, :cx]
                probs_sum[:, z0:z0+cz, y0:y0+cy, x0:x0+cx] += probs_tile[:, :cz, :cy, :cx] * w[None]
                count_map[z0:z0+cz, y0:y0+cy, x0:x0+cx] += w

    probs = probs_sum / np.maximum(count_map[None, ...], 1e-6)
    return probs


# ----- Metrics (only used when GT is present) -----
def dice_per_class(pred, target, num_classes=NUM_CLASSES, eps=1e-6):
    """
    Compute per-class Dice on numpy label volumes.
    pred, target: np.uint8 int labels of shape (Z,Y,X)
    """
    dices = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum()
        denom = p.sum() + t.sum() + eps
        dices.append(2.0 * inter / denom)
    return np.array(dices, dtype=np.float32)


# ----- Visualization helpers -----
def _label_cmap(num_classes=NUM_CLASSES):
    """
    Build a fixed color map for [0..num_classes-1].
    0-bg(black), 1-body(red), 2-bone(gold), 3-bladder(royal blue),
    4-rectum(lime), 5-prostate(hot pink).
    """
    base = np.array([
        [0,   0,   0],     # background
        [220, 20,  60],    # body red
        [255, 215, 0],     # bone gold
        [65,  105, 225],   # bladder royal blue
        [50,  205, 50],    # rectum lime green
        [255, 105, 180],   # prostate hot pink
    ], dtype=np.float32) / 255.0
    return base[:num_classes]

def _pick_slice(vol3d, axis="z", which="center"):
    """
    Pick a 2D slice from a 3D volume by axis and index.
    vol3d: (Z,Y,X) numpy
    which: "center" or integer index as string/int
    """
    Z, Y, X = vol3d.shape
    if axis == "z":
        idx = Z//2 if which == "center" else max(0, min(int(which), Z-1))
        return vol3d[idx, :, :], idx
    elif axis == "y":
        idx = Y//2 if which == "center" else max(0, min(int(which), Y-1))
        return vol3d[:, idx, :], idx
    else:  # "x"
        idx = X//2 if which == "center" else max(0, min(int(which), X-1))
        return vol3d[:, :, idx], idx

def _normalize01(img):
    """Robust 1–99 percentile normalization for display."""
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        lo, hi = img.min(), img.max()
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

def _overlay_rgb(mri2d, pred2d, cmap, alpha=0.45):
    """
    Create an RGB overlay: grayscale MRI + colored labels with transparency.
    mri2d: float [0..1], shape (H,W)
    pred2d: uint labels, shape (H,W)
    """
    rgb = np.stack([mri2d, mri2d, mri2d], axis=-1)        # base gray
    color = cmap[pred2d]                                  # (H,W,3)
    mask = (pred2d != 0).astype(np.float32)[..., None]    # ignore pure background
    return rgb * (1 - alpha * mask) + color * (alpha * mask)

def visualize_and_save(cases, outdir, axis="z", which="center"):
    """
    Draw two figures:
      - Side-by-side: MRI / (GT) / Pred  (3 cols if all have GT, else 2 cols)
      - Overlay:      MRI (row 1) and colored overlay + red prostate contour (row 2)
    'cases' is a list of dicts with keys:
        pid, img:(Z,Y,X) float, lab:(Z,Y,X) or None, pred:(Z,Y,X) uint8, probs:(C,Z,Y,X) float
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    cmap_arr = _label_cmap()
    cmap_listed = ListedColormap(cmap_arr)
    num = len(cases)

    # Decide whether to include GT column: only if ALL have GT with matching shape
    all_have_gt = all(
        (c.get("lab", None) is not None)
        and isinstance(c["lab"], np.ndarray)
        and c["lab"].size > 0
        and c["lab"].shape == c["pred"].shape
        for c in cases
    )
    cols = 3 if all_have_gt else 2

    # ---- Figure 1: Side-by-side
    fig1, axes1 = plt.subplots(num, cols, figsize=(4 * cols, 4 * num))
    if num == 1:
        axes1 = np.expand_dims(axes1, 0)
    if cols == 2 and axes1.ndim == 1:
        axes1 = axes1.reshape(1, 2)

    for i, c in enumerate(cases):
        mri2d, _ = _pick_slice(c["img"], axis, which)
        pred2d, _ = _pick_slice(c["pred"], axis, which)
        mri2d = _normalize01(mri2d)

        # MRI
        axes1[i, 0].imshow(mri2d, cmap="gray")
        axes1[i, 0].set_title(f"Sample {i+1}: MRI Input")
        axes1[i, 0].axis("off")

        next_col = 1
        # Optional GT
        if all_have_gt:
            gt2d, _ = _pick_slice(c["lab"], axis, which)
            axes1[i, 1].imshow(gt2d, vmin=0, vmax=NUM_CLASSES-1, cmap=cmap_listed)
            axes1[i, 1].set_title("Ground Truth (argmax)")
            axes1[i, 1].axis("off")
            next_col = 2

        # Pred
        axes1[i, next_col].imshow(pred2d, vmin=0, vmax=NUM_CLASSES-1, cmap=cmap_listed)
        axes1[i, next_col].set_title("Prediction (argmax)")
        axes1[i, next_col].axis("off")

    plt.tight_layout()
    p1 = Path(outdir) / "predictions_side_by_side.png"
    fig1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"[viz] Saved {p1}")

    # ---- Figure 2: Overlay (always available)
    fig2, axes2 = plt.subplots(2, num, figsize=(4 * num, 8))
    if num == 1:
        axes2 = np.expand_dims(axes2, -1)

    for i, c in enumerate(cases):
        mri2d, _ = _pick_slice(c["img"], axis, which)
        pred2d, _ = _pick_slice(c["pred"], axis, which)
        mri2d = _normalize01(mri2d)
        overlay = _overlay_rgb(mri2d, pred2d, cmap_arr)

        # Row 1: MRI
        axes2[0, i].imshow(mri2d, cmap="gray")
        axes2[0, i].set_title(f"Sample {i+1}: MRI")
        axes2[0, i].axis("off")

        # Row 2: overlay + red contour for prostate
        axes2[1, i].imshow(overlay)
        prostate2d = (pred2d == PROSTATE_IDX).astype(np.float32)
        if prostate2d.any():
            axes2[1, i].contour(prostate2d, levels=[0.5], colors="r", linewidths=1.5)
        axes2[1, i].set_title("Segmentation Overlay (prostate outline in red)")
        axes2[1, i].axis("off")

    plt.tight_layout()
    p2 = Path(outdir) / "predictions_overlay.png"
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[viz] Saved {p2}")


# ----- Main pipeline -----
def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint to rebuild model
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"]
    num_classes = state["outc.weight"].shape[0]
    base = ckpt.get("args", {}).get("base", 16)

    model = UNet3D(in_ch=1, num_classes=num_classes, base=base).to(device)
    model.load_state_dict(state)
    model.eval()

    # Dataset split (train/val/test)
    ds = Prostate3DDataset(args.data_root, split=args.split)

    patch = tuple(args.patch)
    print(f"[INFO] Device={device}, patch={patch}, overlap={args.overlap}, classes={num_classes}, split={args.split}")

    cases_for_fig = []
    all_dices = []       # Collect each case by category Dice
    case_ids = []        # Corresponding cases id

    for i in range(len(ds)):
        s = ds[i]
        pid = s["id"]
        img_t = s["image"].unsqueeze(0).to(device)  # (1,1,Z,Y,X)
        affine = s["affine"]

        # Robustly read label (may be missing for test split)
        lab_np = None
        if "label" in s and s["label"] is not None:
            lab_raw = s["label"]
            if hasattr(lab_raw, "numel"):            # torch.Tensor
                if lab_raw.numel() > 0:
                    lab_np = lab_raw.numpy().astype(np.uint8)
            elif isinstance(lab_raw, np.ndarray) and lab_raw.size > 0:
                lab_np = lab_raw.astype(np.uint8)

        # Probabilities and prediction
        probs = sliding_window_predict_probs(
            model, img_t, num_classes=num_classes,
            patch=patch, overlap=args.overlap, device=device, amp=args.amp
        )
        pred = probs.argmax(0).astype(np.uint8)      # (Z,Y,X)

        # Save NIfTI
        out_path = Path(args.outdir) / f"{pid}_pred.nii.gz"
        nib.save(nib.Nifti1Image(pred, affine=affine), str(out_path))
        print(f"[save] {out_path}")

        # Dice (only if GT exists and shape matches)
        if (lab_np is not None) and (lab_np.shape == pred.shape):
            dices = dice_per_class(pred, lab_np, num_classes=num_classes)
            log_line = " | ".join([f"{CLASS_NAMES[c]}: {dices[c]:.4f}" for c in range(num_classes)])
            print(f"[dice] {pid} -> {log_line}")
            all_dices.append(dices)
            case_ids.append(pid)

        else:
            print(f"[dice] {pid} -> skip (no GT or shape mismatch). pred={pred.shape}, "
                  f"lab={'None' if lab_np is None else lab_np.shape}")

        # Collect samples for figures
        if len(cases_for_fig) < args.num_samples:
            img_np = s["image"].numpy()[0]  # (Z,Y,X) float
            cases_for_fig.append({
                "pid": pid,
                "img": img_np,
                "lab": lab_np,
                "pred": pred,
                "probs": probs
            })
    
    # ===== Dataset-level summary =====
    if all_dices:
        D = np.stack(all_dices, axis=0)           # (N, C)
        mean_per_class = D.mean(axis=0)           # (C,)
        mean_all = float(mean_per_class.mean())   # include bg
        mean_org = float(mean_per_class[1:].mean())  # no bg

        print("\n[summary] Mean Dice per class (dataset):")
        for c in range(num_classes):
            print(f"  {CLASS_NAMES[c]:<10s}: {mean_per_class[c]:.4f}")
        print(f"[summary] Mean Dice (all classes)   : {mean_all:.4f}")
        print(f"[summary] Mean Dice (organs only)   : {mean_org:.4f}")

        # Save CSV
        csv_path = Path(args.outdir) / f"{args.split}_dice_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case"] + CLASS_NAMES)
            for pid, d in zip(case_ids, all_dices):
                w.writerow([pid] + [f"{x:.4f}" for x in d])
            w.writerow(["MEAN(all cases)"] + [f"{x:.4f}" for x in mean_per_class])
            w.writerow(["MEAN_all_classes", f"{mean_all:.4f}"])
            w.writerow(["MEAN_organs_only", f"{mean_org:.4f}"])
        print(f"[summary] Saved CSV -> {csv_path}\n")

    # Draw figures
    if cases_for_fig:
        visualize_and_save(
            cases_for_fig, args.outdir,
            axis=args.axis.lower(), which=(args.slice if args.slice != "center" else "center")
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Root dir containing semantic_MRs_anon / semantic_labels_anon")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to run on")
    ap.add_argument("--ckpt", type=str, default="runs/best.ckpt", help="Path to checkpoint saved by train.py")
    ap.add_argument("--outdir", type=str, default="runs/preds", help="Output directory for NIfTI and figures")
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--patch", type=int, nargs=3, default=[64,64,64], help="Sliding-window patch size (Z Y X)")
    ap.add_argument("--overlap", type=int, default=16, help="Sliding-window overlap on each axis")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision during inference")
    ap.add_argument("--num_samples", type=int, default=4, help="Number of cases to visualize")
    ap.add_argument("--axis", type=str, default="z", choices=["z", "y", "x"], help="Axis used for 2D figures")
    ap.add_argument("--slice", type=str, default="center", help="'center' or an integer slice index")
    args = ap.parse_args()
    main(args)
