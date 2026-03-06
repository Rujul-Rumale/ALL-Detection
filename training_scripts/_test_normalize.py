"""Quick diagnostic: test sequential + threaded normalization on small sample."""
import os, time, numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

TRAIN_BASE = r'c:\Open Source\leukiemea\C-NMC_Dataset\PKG - C-NMC 2019\C-NMC_training_data'
OUT_TRAIN  = r'c:\Open Source\leukiemea\cnmc_normed_training'

class MacenkoNormalizer:
    def __init__(self):
        self.HERef   = np.array([[0.5626, 0.2159],[0.7201, 0.8012],[0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])
    def normalize(self, img):
        h, w, _ = img.shape
        I = img.reshape(-1, 3).astype(np.float64)
        OD = -np.log10(np.clip(I/255.0, 1e-6, 1.0))
        mask = np.all(OD > 0.15, axis=1)
        if mask.sum() < 10:
            HE, maxC = self.HERef, self.maxCRef
        else:
            _, _, V = np.linalg.svd(OD[mask], full_matrices=False)
            V = V[:2, :]
            proj = OD[mask] @ V.T
            phi = np.arctan2(proj[:, 1], proj[:, 0])
            v1 = V.T @ np.array([np.cos(np.percentile(phi, 1)), np.sin(np.percentile(phi, 1))])
            v2 = V.T @ np.array([np.cos(np.percentile(phi, 99)), np.sin(np.percentile(phi, 99))])
            HE = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
            maxC = np.percentile(OD[mask] @ np.linalg.pinv(HE), 99, axis=0)
        C = (OD @ np.linalg.pinv(HE)) / maxC * self.maxCRef
        return np.clip(255 * np.power(10, -(C @ self.HERef.T)), 0, 255).astype(np.uint8).reshape(h, w, 3)

def worker(args):
    src, dst = args
    try:
        n = MacenkoNormalizer()
        img = np.array(Image.open(src).convert('RGB'))
        out = n.normalize(img)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Image.fromarray(out).save(dst)
        return True, src, None
    except Exception as e:
        return False, src, str(e)

# Build small task list
tasks = []
for fold in ['fold_0', 'fold_1', 'fold_2']:
    for cls in ['all', 'hem']:
        d = os.path.join(TRAIN_BASE, fold, cls)
        for f in list(os.listdir(d))[:3]:
            src = os.path.join(d, f)
            dst = os.path.join(OUT_TRAIN, fold, cls, os.path.splitext(f)[0] + '.png')
            tasks.append((src, dst))

print(f"Testing {len(tasks)} images sequentially...")
t0 = time.time()
for t in tasks:
    ok, src, err = worker(t)
    status = "OK" if ok else "FAIL"
    print(f"  {status}: {os.path.basename(src)}" + (f" | {err}" if err else ""))
print(f"Sequential done in {time.time()-t0:.1f}s")

# Clean for thread test
import shutil
if os.path.exists(OUT_TRAIN): shutil.rmtree(OUT_TRAIN)

print(f"\nTesting {len(tasks)} images with 4 threads...")
t0 = time.time()
fails = []
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(worker, t): t for t in tasks}
    for fut in as_completed(futures):
        ok, src, err = fut.result()
        if not ok:
            fails.append((src, err))
print(f"Thread done in {time.time()-t0:.1f}s. Failures: {len(fails)}")
for src, err in fails:
    print(f"  FAIL: {src} | {err}")
