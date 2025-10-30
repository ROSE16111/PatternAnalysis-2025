# quick_check.py —— read a NIfTI，check data and lables
# run: ``````D:/document/UQ/4COMP3710/pytorch_env/python.exe d:/document/UQ/4COMP3710/A3/PatternAnalysis-2025/recognition/prostate3d_Luyi_Ying_48360591/quick_dataCheck.py --data_root "D:\document\UQ\4COMP3710\A3\data"
#print：
# [INFO] Saved split file to D:\document\UQ\4COMP3710\A3\data\splits.json
#ID: Case_013_Week3_LFOV
#Image shape (C,Z,Y,X): (1, 256, 256, 128)
#Label uniques: [0, 1, 2, 3, 4, 5]
import argparse, torch
from dataset import Prostate3DDataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="dir includs semantic_MRs_anon / semantic_labels_anon")
    args = ap.parse_args()

    ds = Prostate3DDataset(args.data_root, split="train")
    s = ds[0]
    img, lab = s["image"], s["label"]
    print("ID:", s["id"])
    print("Image shape (C,Z,Y,X):", tuple(img.shape))
    if lab.numel() > 0:
        print("Label uniques:", torch.unique(lab).tolist())  # expect [0,1,2,3,4,5]
    else:
        print("No label (test split)")
