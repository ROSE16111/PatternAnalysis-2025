# train.py —— 3D UNet 基线（Normal）：训练并在验证集打印“逐类 Dice”
r"""
to run smock test
python recognition\prostate3d_Luyi_Ying_48360591\train.py --data_root "D:\document\UQ\4COMP3710\A3\data" --epochs 2 --batch_size 1 --base 16 --amp
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

    for epoch in range(1, args.epochs + 1):
        # ---------------- 训练 ----------------
        model.train()
        for batch in dl_train:
            img = batch["image"].to(device)   # (B,1,Z,Y,X)
            lab = batch["label"].to(device)   # (B,Z,Y,X)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(img)           # (B,C,Z,Y,X)
                loss = criterion(logits, lab)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ---------------- 验证 ----------------
        model.eval()
        dices_all = []
        with torch.no_grad():
            for batch in dl_val:
                img = batch["image"].to(device)
                lab = batch["label"].to(device)
                logits = model(img)
                dices_all.append(dice_per_class(logits, lab))  # shape=(C,)

        if len(dices_all):
            dices_all = np.stack(dices_all)        # (Nval, C)
            per_class = dices_all.mean(0)          # (C,)
            mdice = float(per_class.mean())        # mean over classes
        else:
            per_class = np.zeros(NUM_CLASSES, dtype=np.float32)
            mdice = 0.0

        # 打印逐类 Dice 与是否达标
        print(f"\nEpoch {epoch:03d} | mean Dice (val): {mdice:.4f}")
        for name, v in zip(CLASS_NAMES, per_class):
            status = "OK" if v >= 0.70 else "LOW"
            print(f"  - {name:<8s}: {v:.4f}  [{status}]")
        print("")

        # 保存最好模型（按 mean Dice）
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
    args = ap.parse_args()
    main(args)
