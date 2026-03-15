"""
Pre-compute Macenko Stain Normalization for C-NMC Dataset
=========================================================
Normalizes all images once and saves to cnmc_staging_normed/.
Training then uses the pre-normalized images directly (no on-the-fly overhead).

Usage:
  python training_scripts/precompute_stain_norm.py
"""
import os
import sys
import time
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import PROJECT_ROOT

STAGING_DIR = str(PROJECT_ROOT / "cnmc_staging")
OUTPUT_DIR = str(PROJECT_ROOT / "cnmc_staging_normed")

# ============ MACENKO NORMALIZER ============
class MacenkoNormalizer:
    def __init__(self):
        self.HERef = np.array([[0.5626, 0.2159],
                                [0.7201, 0.8012],
                                [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def _get_stain_matrix(self, I, beta=0.15, alpha=1):
        I = I.reshape(-1, 3).astype(np.float64)
        OD = -np.log10(np.clip(I / 255.0, 1e-6, 1.0))
        mask = np.all(OD > beta, axis=1)
        if mask.sum() < 10:
            return self.HERef, self.maxCRef
        OD_hat = OD[mask]
        try:
            _, _, V = np.linalg.svd(OD_hat, full_matrices=False)
        except np.linalg.LinAlgError:
            return self.HERef, self.maxCRef
        V = V[:2, :]
        proj = OD_hat @ V.T
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        v1 = V.T @ np.array([np.cos(minPhi), np.sin(minPhi)])
        v2 = V.T @ np.array([np.cos(maxPhi), np.sin(maxPhi)])
        if v1[0] > v2[0]:
            HE = np.array([v1, v2]).T
        else:
            HE = np.array([v2, v1]).T
        Y = OD @ np.linalg.pinv(HE)
        maxC = np.percentile(Y, 99, axis=0)
        return HE, maxC

    def normalize(self, img):
        h, w, _ = img.shape
        try:
            HE, maxC = self._get_stain_matrix(img)
        except Exception:
            return img
        OD = -np.log10(np.clip(img.reshape(-1, 3).astype(np.float64) / 255.0, 1e-6, 1.0))
        C = OD @ np.linalg.pinv(HE)
        C = C / maxC * self.maxCRef
        OD_norm = C @ self.HERef.T
        I_norm = np.clip(255.0 * np.power(10, -OD_norm), 0, 255).astype(np.uint8)
        return I_norm.reshape(h, w, 3)


def process_image(args):
    """Process a single image (for multiprocessing)."""
    src_path, dst_path, normalizer = args
    try:
        img = np.array(Image.open(src_path).convert("RGB"))
        normed = normalizer.normalize(img)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        Image.fromarray(normed).save(dst_path, quality=95)
        return True, src_path
    except Exception as e:
        # Copy original on failure
        try:
            Image.open(src_path).save(dst_path, quality=95)
        except:
            pass
        return False, f"{src_path}: {e}"


def main():
    normalizer = MacenkoNormalizer()
    
    # Collect all images
    tasks = []
    for split in ["train", "val"]:
        split_dir = os.path.join(STAGING_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            out_cls_dir = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(out_cls_dir, exist_ok=True)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    src = os.path.join(cls_dir, fname)
                    # Save as PNG to avoid JPEG re-compression artifacts
                    dst = os.path.join(out_cls_dir, os.path.splitext(fname)[0] + ".png")
                    tasks.append((src, dst, normalizer))

    total = len(tasks)
    print(f"Stain-normalizing {total} images...")
    print(f"  Source: {STAGING_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    t0 = time.time()
    done = 0
    failed = 0

    # Process with multiprocessing for speed
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(process_image, t) for t in tasks]
        for f in as_completed(futures):
            success, info = f.result()
            done += 1
            if not success:
                failed += 1
            if done % 500 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done:>6d}/{total}] {rate:.1f} img/s | "
                      f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s | "
                      f"failed {failed}")

    elapsed = time.time() - t0
    print(f"\nDone! {total} images in {elapsed:.0f}s ({total/elapsed:.1f} img/s)")
    print(f"  Failed: {failed}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"\nTo train with pre-normalized data, update STAGING_DIR in train_cnmc_large_v3.py")
    print(f"  or run: python training_scripts/train_cnmc_large_v3.py --no_stain_norm")
    print(f"  (the images are already normalized, so disable on-the-fly normalization)")


if __name__ == "__main__":
    main()
