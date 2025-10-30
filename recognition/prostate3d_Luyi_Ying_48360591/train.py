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

# 正式跑 30 个 epoch（patch 训练 + AMP）
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python recognition\prostate3d_Luyi_Ying_48360591\train.py `
  --data_root "D:\document\UQ\4COMP3710\A3\data" `
  --epochs 30 --batch_size 1 `
  --base 8 `
  --patch 64 64 64 `
  --accum 1 `
  --amp

  
python recognition\prostate3d_Luyi_Ying_48360591\train.py `
  --data_root "D:\document\UQ\4COMP3710\A3\data" `
  --epochs 30 --batch_size 1 `
  --base 8 --patch 64 64 64 --accum 1 --amp `
  --fullval_every 5 --val_patch 64 64 64 --val_overlap 16

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
import torch.nn.functional as F

def _starts(L, P, O):
    if L <= P: return [0]
    stride = max(P - O, 1)
    s = list(range(0, L - P + 1, stride))
    if s[-1] != L - P: s.append(L - P)
    return s

@torch.no_grad()
def sliding_window_predict_logits(model, img, num_classes, patch, overlap, device, amp=True):
    # img: (B=1, C=1, Z,Y,X) on device
    _, _, Z, Y, X = img.shape
    pz, py, px = patch
    sz, sy, sx = _starts(Z,pz,overlap), _starts(Y,py,overlap), _starts(X,px,overlap)
    probs_sum = np.zeros((num_classes, Z, Y, X), dtype=np.float32)
    count_map = np.zeros((Z, Y, X), dtype=np.float32)

    autocast = torch.cuda.amp.autocast if device.type=="cuda" else torch.cpu.amp.autocast
    model.eval()
    for z0 in sz:
        for y0 in sy:
            for x0 in sx:
                tile = img[:, :, z0:z0+pz, y0:y0+py, x0:x0+px]
                with autocast(enabled=amp):
                    logits = model(tile)                  # (1,C,*,*,*)
                    probs  = F.softmax(logits, dim=1)[0].float().cpu().numpy()
                cz, cy, cx = probs.shape[1:]
                probs_sum[:, z0:z0+cz, y0:y0+cy, x0:x0+cx] += probs
                count_map[z0:z0+cz, y0:y0+cy, x0:x0+cx] += 1
    probs = probs_sum / np.maximum(count_map[None,...], 1e-6)
    return probs  # (C,Z,Y,X)

# (For val dataloader, calculate per-class Dice)
def full_volume_validation(model, dl_val, device, num_classes=5, patch=(64,64,64), overlap=16, amp=True):
    model.eval()
    dices_all = []
    with torch.no_grad():
        for batch in dl_val:
            img = batch["image"].to(device)  # 形状已是 (1,1,Z,Y,X)
            lab = batch["label"].to(device)  # 形状已是 (1,Z,Y,X)
            probs = sliding_window_predict_logits(model, img, num_classes, patch, overlap, device, amp)
            pred  = probs.argmax(0)  # (Z,Y,X)
            per_c = []
            for c in range(num_classes):
                p = (pred == c); t = (lab[0] == c)
                inter = (p & t).sum().item()
                denom = p.sum().item() + t.sum().item() + 1e-6
                per_c.append(2.0*inter/denom)
            dices_all.append(per_c)
    dices_all = np.array(dices_all, dtype=np.float32) if len(dices_all) else np.zeros((1,num_classes), np.float32)
    return dices_all.mean(0)  # per-class Dice

def dice_loss_multiclass(logits, target, eps=1e-5):
    """
    logits: (B,C,Z,Y,X), target: (B,Z,Y,X) in [0..C-1]
    """
    C = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=C).permute(0,4,1,2,3).float()
    dims = (0,2,3,4)
    inter = (probs * onehot).sum(dims)
    denom = (probs*probs).sum(dims) + (onehot*onehot).sum(dims)
    dice = (2*inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

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

            # ----- 轻量增强light aug -----
            if torch.rand(1).item() < 0.5:  # 随机翻转三个轴
                if torch.rand(1).item() < 0.5:
                    img_c = img_c.flip(-1); lab_c = lab_c.flip(-1)  # X
                if torch.rand(1).item() < 0.5:
                    img_c = img_c.flip(-2); lab_c = lab_c.flip(-2)  # Y
                if torch.rand(1).item() < 0.5:
                    img_c = img_c.flip(-3); lab_c = lab_c.flip(-3)  # Z
            # 轻度强度扰动（亮度/对比度 + 微噪声）
            if torch.rand(1).item() < 0.5:
                scale = 1.0 + 0.10*torch.randn((), device=img_c.device)   # ±10%
                shift = 0.05*torch.randn((), device=img_c.device)          # ±0.05
                img_c = img_c*scale + shift
                img_c = img_c.clamp(-5, 5)
            # ----------------------

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(img_c)
                ce = F.cross_entropy(logits, lab_c)
                dl = dice_loss_multiclass(logits, lab_c)
                loss = ce + 0.5*dl
            loss = loss / accum


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

        # ---- 周期性全幅滑窗验证（更客观，用它来挑 best）----
        do_full = (epoch % args.fullval_every == 0) or (epoch == args.epochs)
        if do_full:
            per_class_full = full_volume_validation(
                model, dl_val, device, num_classes=5,
                patch=tuple(args.val_patch), overlap=args.val_overlap, amp=args.amp
            )
            mdice_full = float(per_class_full.mean())
            print(f"[FULLVAL] mean Dice: {mdice_full:.4f}")
            for name, v in zip(["body","bone","bladder","rectum","prostate"], per_class_full):
                print(f"  - {name:<8s}: {v:.4f}  [{'OK' if v>=0.70 else 'LOW'}]")

            # 用“全幅指标”决定是否保存 best
            if mdice_full > best_mdice:
                best_mdice = mdice_full
                torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
                print(f"[SAVE] best(full) -> {ckpt_path} (mean Dice={best_mdice:.4f})")
        else:
            # 如果本 epoch 不做全幅，用中心-patch 指标兜底挑 best（可选）
            if mdice > best_mdice:
                best_mdice = mdice
                torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
                print(f"[SAVE] best(center) -> {ckpt_path} (mean Dice={best_mdice:.4f})")


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
    ap.add_argument("--fullval_every", type=int, default=5, help="每 N 个 epoch 跑一次全幅滑窗验证")
    ap.add_argument("--val_patch", type=int, nargs=3, default=[64,64,64], help="验证/滑窗 patch")
    ap.add_argument("--val_overlap", type=int, default=16, help="滑窗重叠")

    args = ap.parse_args()
    main(args)
