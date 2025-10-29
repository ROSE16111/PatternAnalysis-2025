# train.py —— 3D UNet 基线（Normal）：训练并在验证集打印“逐类 Dice”
r"""
to run smock test

$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python recognition\prostate3d_Luyi_Ying_48360591\train.py `
  --data_root "D:\document\UQ\4COMP3710\A3\data" `
  --epochs 2 --batch_size 1 `
  --base 8 `
  --patch 64 64 64 `
  --amp

base:Smaller model, saves video memory; patch:64³ patch; amp: mixed precision
Seeing the class-specific Dice for each epoch and generating runs\best.ckpt indicates that the training pipeline is OK.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import UNet3D
from dataset import Prostate3DDataset
import math

def random_crop_3d(img, lab, size):
    """
    Randomly select a patch from the 3D volume.
    img: (B,1,Z,Y,X) float32
    lab: (B,Z,Y,X)   int64
    size: (pz, py, px)
    Returns a small block of the same dtype/device.
    Reduce memory pressure
    """
    B, C, Z, Y, X = img.shape
    pz, py, px = size
    # 保证不越界（如果原图更小就从 0 开始并在后面做必要的 pad）
    z0 = 0 if Z <= pz else torch.randint(0, Z - pz + 1, (1,), device=img.device).item()
    y0 = 0 if Y <= py else torch.randint(0, Y - py + 1, (1,), device=img.device).item()
    x0 = 0 if X <= px else torch.randint(0, X - px + 1, (1,), device=img.device).item()
    z1, y1, x1 = min(z0+pz, Z), min(y0+py, Y), min(x0+px, X)
    img_c = img[:, :, z0:z1, y0:y1, x0:x1]
    lab_c = lab[:,    z0:z1, y0:y1, x0:x1]
    # 若边界导致尺寸比目标小（极少数情况），做零填充到目标 size
    if img_c.shape[2:] != (pz, py, px):
        pad_z = pz - img_c.shape[2]
        pad_y = py - img_c.shape[3]
        pad_x = px - img_c.shape[4]
        img_c = torch.nn.functional.pad(img_c, (0, max(pad_x,0), 0, max(pad_y,0), 0, max(pad_z,0)))
        lab_c = torch.nn.functional.pad(lab_c, (0, max(pad_x,0), 0, max(pad_y,0), 0, max(pad_z,0)))
        img_c = img_c[:, :, :pz, :py, :px]
        lab_c = lab_c[:,    :pz, :py, :px]
    return img_c, lab_c

def dice_per_class_from_logits(logits, target, num_classes=5):
    with torch.no_grad():
        pred = logits.argmax(1)  # (B,Z,Y,X)
        dices = []
        for c in range(num_classes):
            p = (pred == c)
            t = (target == c)
            inter = (p & t).sum().item()
            denom = p.sum().item() + t.sum().item() + 1e-6
            dices.append(2.0 * inter / denom)
        return np.array(dices, dtype=np.float32)

NUM_CLASSES = 5
CLASS_NAMES = ["body", "bone", "bladder", "rectum", "prostate"]

def dice_per_class(logits, target, num_classes=NUM_CLASSES):
    """计算逐类 Dice（per-class Dice）"""
    with torch.no_grad():
        pred = logits.argmax(1)  # (B,Z,Y,X)
        dices = []
        for c in range(num_classes):
            p = (pred == c)
            t = (target == c)
            inter = (p & t).sum().item()
            denom = p.sum().item() + t.sum().item() + 1e-6
            dices.append(2.0 * inter / denom)
        return np.array(dices, dtype=np.float32)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device = {device}")

    # 数据加载（首次运行会在 <data_root>/splits.json 生成数据划分）
    ds_train = Prostate3DDataset(args.data_root, split="train")
    ds_val   = Prostate3DDataset(args.data_root, split="val")
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=1, shuffle=False, num_workers=0)

    # 模型（UNet3D baseline）
    model = UNet3D(in_ch=1, num_classes=NUM_CLASSES, base=args.base).to(device)

    # 损失 + 优化器（baseline：CE，可日后换 DiceLoss/组合以提升）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_mdice = 0.0
    outdir = Path("runs"); outdir.mkdir(exist_ok=True)
    ckpt_path = outdir / "best.ckpt"

    patch = tuple(args.patch)
    accum = max(1, args.accum)

    for epoch in range(1, args.epochs + 1):
        # -------- 训练（patch-based）--------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_in_accum = 0

        for batch in dl_train:
            img = batch["image"].to(device)   # (B,1,Z,Y,X) 这里 B=1
            lab = batch["label"].to(device)   # (B,Z,Y,X)

            # 从整幅里裁一个 patch
            img_c, lab_c = random_crop_3d(img, lab, patch)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(img_c)
                loss = criterion(logits, lab_c) / accum

            scaler.scale(loss).backward()
            step_in_accum += 1
            if step_in_accum == accum:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step_in_accum = 0

        # 如果最后不足 accum 也要 step 一下
        if step_in_accum > 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

        # -------- 验证（用较小中心区域评估，避免 OOM）--------
        model.eval()
        dices_all = []
        with torch.no_grad():
            for batch in dl_val:
                img = batch["image"].to(device)   # (B,1,Z,Y,X)
                lab = batch["label"].to(device)   # (B,Z,Y,X)
                # 选中心区域的 patch 做验证（简易；后期可换滑窗全幅）
                B, C, Z, Y, X = img.shape
                pz, py, px = patch
                z0 = max(0, (Z - pz)//2); y0 = max(0, (Y - py)//2); x0 = max(0, (X - px)//2)
                z1, y1, x1 = min(z0+pz, Z), min(y0+py, Y), min(x0+px, X)
                img_c = img[:, :, z0:z1, y0:y1, x0:x1]
                lab_c = lab[:,    z0:z1, y0:y1, x0:x1]

                logits = model(img_c)
                dices_all.append(dice_per_class_from_logits(logits, lab_c, num_classes=5))

        if len(dices_all):
            dices_all = np.stack(dices_all); per_class = dices_all.mean(0); mdice = float(per_class.mean())
        else:
            per_class = np.zeros(5, dtype=np.float32); mdice = 0.0

        print(f"\nEpoch {epoch:03d} | mean Dice (val, center-patch): {mdice:.4f}")
        for name, v in zip(["body","bone","bladder","rectum","prostate"], per_class):
            print(f"  - {name:<8s}: {v:.4f}  [{'OK' if v>=0.70 else 'LOW'}]")
        print("")

        if mdice > best_mdice:
            best_mdice = mdice
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"[SAVE] best -> {ckpt_path} (mean Dice={best_mdice:.4f})")

    print("[INFO] Training finished.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="含 semantic_MRs_anon / semantic_labels_anon 的目录")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1, help="3D 体分割通常 batch 很小")
    ap.add_argument("--base", type=int, default=16, help="UNet 基通道；显存吃紧可用 16/24/32")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true", help="混合精度，省显存更快")
    ap.add_argument("--patch", type=int, nargs=3, default=[64,64,64], help="3D patch size (Z Y X)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    args = ap.parse_args()
    main(args)
