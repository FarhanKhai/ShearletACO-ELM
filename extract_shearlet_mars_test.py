import os
import numpy as np
import imageio.v2 as iio
from FFST import shearletTransformSpect

# ==========================
# 1. PARAMETER DATASET (TEST)
# ==========================
DATA_ROOT   = "."
LABEL_FILE  = os.path.join(DATA_ROOT, "test-calibrated-shuffled.txt") # BACA FILE TEST
OUTPUT_NPZ  = os.path.join(DATA_ROOT, "shearlet_test.npz")            # OUTPUT TEST

# ==========================
# 2. NORMALISASI UKURAN
# ==========================
def normalize_to_256(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2] 
    
    # Crop tengah
    if h > 256:
        top = (h - 256) // 2
        img = img[top:top+256, ...]
    if w > 256:
        left = (w - 256) // 2
        img = img[:, left:left+256, ...]
        
    # Pad
    h, w = img.shape[:2]
    pad_top = max((256 - h) // 2, 0)
    pad_bottom = 256 - h - pad_top
    pad_left = max((256 - w) // 2, 0)
    pad_right = 256 - w - pad_left
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        if img.ndim == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode="edge")
        else:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
            
    return img

# ==========================
# 3. EKSTRAKSI FITUR (WARNA + SHEARLET)
# ==========================
def extract_color_features(img_rgb: np.ndarray) -> np.ndarray:
    """Mengambil Mean dan Std Dev dari R, G, B (Total 6 fitur)"""
    means = np.mean(img_rgb, axis=(0, 1)) # [Mean_R, Mean_G, Mean_B]
    stds  = np.std(img_rgb, axis=(0, 1))  # [Std_R, Std_G, Std_B]
    return np.concatenate([means, stds]).astype(np.float32)

def extract_shearlet_features(img_gray: np.ndarray) -> np.ndarray:
    ST, Psi = shearletTransformSpect(img_gray)
    feats = []
    for k in range(ST.shape[2]):
        band = ST[:, :, k]
        mag  = np.abs(band)
        feats.extend([np.sum(mag**2), np.mean(mag), np.var(mag)])
    return np.array(feats, dtype=np.float32)

def load_image_label_pairs(label_file):
    pairs = []
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rel_path, label_str = line.split()
            full_path = os.path.join(DATA_ROOT, rel_path)
            pairs.append((full_path, int(label_str)))
    return pairs

def main():
    if not os.path.exists(LABEL_FILE):
        print(f"File label tidak ditemukan: {LABEL_FILE}")
        return

    pairs = load_image_label_pairs(LABEL_FILE)
    print(f"Total TEST samples: {len(pairs)}")
    X_list, y_list = [], []

    for i, (img_path, label) in enumerate(pairs, start=1):
        if not os.path.exists(img_path): continue
        try:
            img = iio.imread(img_path) 
            img = normalize_to_256(img)

            # Ekstrak Warna & Convert ke Gray
            feat_color = np.zeros(6, dtype=np.float32)
            if img.ndim == 3:
                feat_color = extract_color_features(img)
                img_gray = img.mean(axis=2).astype(np.float32)
            else:
                img_gray = img.astype(np.float32)

            feat_shearlet = extract_shearlet_features(img_gray)
            full_feats = np.concatenate([feat_shearlet, feat_color])

            X_list.append(full_feats)
            y_list.append(label)

        except Exception as e:
            print(f"Error {img_path}: {e}")
            continue

        if i % 50 == 0: print(f"Processed {i}/{len(pairs)}")

    if not X_list: return
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    np.savez(OUTPUT_NPZ, X=X, y=y)
    print("Saved TEST features to:", OUTPUT_NPZ)

if __name__ == "__main__":
    main()