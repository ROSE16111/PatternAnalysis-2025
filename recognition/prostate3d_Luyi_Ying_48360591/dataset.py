# dataset.py —— canonicalize and maps MR 与 Label，Compatible with *_anon naming differences (e.g., _SEMANTIC)
from pathlib import Path
import re, json, random
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

def load_nii(path):
    """read .nii/.nii.gz，return (data, affine). if 4D(...,1) then zip it as 3D。"""
    img = nib.load(str(path))
    arr = img.get_fdata(caching='unchanged').astype(np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr, img.affine

def _find_dir(root: Path, candidates):
    """find dir"""
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {candidates} found under {root}")

_ext_re = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)

def _strip_ext(name: str) -> str:
    """Remove the .nii or .nii.gz extension (string processing only, no file reading)"""
    return _ext_re.sub("", name)

def _canonical_id(filename: str) -> str:
    """
    Normalize IDs: Remove the extension, remove '_SEMANTIC', and merge duplicate underscores into one.
    e.g.'Case_004_Week0_SEMANTIC_LFOV.nii.gz' -> 'Case_004_Week0_LFOV'
    """
    base = _strip_ext(filename)
    base = base.replace("_SEMANTIC", "")
    base = re.sub(r"_+", "_", base)  # Merge consecutive underscores
    return base

class Prostate3DDataset(Dataset):
    """
    structure：
    <root>/
      semantic_MRs_anon/ or semantic_MRs/
      semantic_labels_anon/ or semantic_labels_only/
    On the first run, an 80/10/10 split will be automatically generated in <root>/splits.json.
    """
    def __init__(self, root, split="train", transforms=None, split_json="splits.json", seed=42):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        self.img_dir = _find_dir(self.root, ["semantic_MRs_anon", "semantic_MRs"])
        self.lab_dir = _find_dir(self.root, ["semantic_labels_anon", "semantic_labels_only"])

        # Read all files in two directories and create {canonical_id: Path} map
        img_files = sorted(self.img_dir.glob("*.nii*"))
        lab_files = sorted(self.lab_dir.glob("*.nii*"))

        self.img_index = {_canonical_id(p.name): p for p in img_files}
        self.lab_index = {_canonical_id(p.name): p for p in lab_files}

        # Only keep samples that exist on both sides
        common_ids = sorted(set(self.img_index.keys()) & set(self.lab_index.keys()))
        if len(common_ids) == 0:
            # Print some samples to help with the investigation.
            ex_img = [p.name for p in img_files[:3]]
            ex_lab = [p.name for p in lab_files[:3]]
            raise RuntimeError(
                "No matching image/label pairs found.\n"
                f"Examples images: {ex_img}\nExamples labels: {ex_lab}\n"
                "Hint: filename patterns differ a lot? Check _canonical_id() rule."
            )

        # Read/Generate Partitions
        self.split_file = self.root / split_json
        if self.split_file.exists():
            with open(self.split_file, "r", encoding="utf-8") as f:
                sp = json.load(f)
        else:
            ids = common_ids.copy()
            random.Random(seed).shuffle(ids)
            n = len(ids)
            n_train = int(0.8 * n)
            n_val   = max(1, int(0.1 * n))
            sp = {
                "train": ids[:n_train],
                "val":   ids[n_train:n_train+n_val],
                "test":  ids[n_train+n_val:]
            }
            with open(self.split_file, "w", encoding="utf-8") as f:
                json.dump(sp, f, indent=2)
            print(f"[INFO] Saved split file to {self.split_file}")

        self.ids = sp[split]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]  # canonical id
        img_path = self.img_index[cid]
        lab_path = self.lab_index[cid] if self.split != "test" else None

        # read img
        img, aff = load_nii(img_path)
        # z-score normalization
        m, s = img.mean(), img.std() + 1e-6
        img = (img - m) / s
        img_t = torch.from_numpy(img[None, ...])  # (1,Z,Y,X)

        # read lable and maps as 0..4
        if lab_path is not None:
            lab, _ = load_nii(lab_path)
            lab = lab.astype(np.int64)
            if lab.min() >= 1:
                lab = lab - 1
            lab = np.clip(lab, 0, 4)
            lab_t = torch.from_numpy(lab)  # (Z,Y,X)
        else:
            lab_t = torch.tensor([])

        sample = {"id": cid, "image": img_t, "label": lab_t, "affine": aff}
        if self.transforms:
            sample = self.transforms(sample)
        return sample
