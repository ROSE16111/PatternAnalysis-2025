# tools_count_labels.py
# python recognition\prostate3d_Luyi_Ying_48360591\tools_count_labels.py

from pathlib import Path
import numpy as np
import nibabel as nib
from dataset import _find_dir

root = Path(r"D:\document\UQ\4COMP3710\A3\data")  # 改成你的data根
lab_dir = _find_dir(root, ["semantic_labels_anon", "semantic_labels_only"])

tot = np.zeros(6, dtype=np.int64)  # 6类：0..5
for p in sorted(lab_dir.glob("*.nii*")):
    arr = nib.load(str(p)).get_fdata(caching='unchanged')
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[...,0]
    arr = arr.astype(np.int64)
    # 如果你的dataset里label是1..6，这里要减1；如果本来就是0..5，就不要减
    # 你先不做减1，直接看unique值
    uniq, cnt = np.unique(arr, return_counts=True)
    print(p.name, dict(zip(uniq.tolist(), cnt.tolist())))
    for u, c in zip(uniq, cnt):
        if 0 <= u < 6:
            tot[u] += int(c)

print("TOTAL per class (label->voxels):", {i:int(tot[i]) for i in range(6)})
