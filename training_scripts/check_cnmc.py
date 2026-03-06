import cv2
import os

img = cv2.imread(r"c:\Open Source\leukiemea\C-NMC\fold_0\fold_0\all\UID_11_1_1_all.bmp")
print(f"Sample Image Shape: {img.shape}")

for fold in ["fold_0", "fold_1", "fold_2"]:
    base = os.path.join(r"c:\Open Source\leukiemea\C-NMC", fold, fold)
    a = len(os.listdir(os.path.join(base, "all")))
    h = len(os.listdir(os.path.join(base, "hem")))
    print(f"{fold}: all={a}, hem={h}")
