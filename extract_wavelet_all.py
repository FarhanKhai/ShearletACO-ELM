import os
import numpy as np
import imageio.v2 as iio
import pywt  # Library Wavelet

# ==========================
# KONFIGURASI
# ==========================
DATASETS = {
    "train": ("train-calibrated-shuffled.txt", "wavelet_train.npz"),
    "val":   ("val-calibrated-shuffled.txt",   "wavelet_val.npz"),
    "test":  ("test-calibrated-shuffled.txt",  "wavelet_test.npz")
}

def extract_color_features(img_rgb):
    # 6 Fitur Warna (Mean & Std)
    means = np.mean(img_rgb, axis=(0, 1))
    stds  = np.std(img_rgb, axis=(0, 1))
    return np.concatenate([means, stds])

def extract_wavelet_features(img_gray):
    # Menggunakan Wavelet 'db1' (Daubechies 1 / Haar) 
    # Level 1 decomposition
    coeffs = pywt.dwt2(img_gray, 'db1')
    LL, (LH, HL, HH) = coeffs

    feats = []
    # Statistik Energy, Mean, Variance dari detail coefficients
    for band in [LH, HL, HH]:
        mag = np.abs(band)
        feats.extend([np.sum(mag**2), np.mean(mag), np.var(mag)])
    
    return np.array(feats, dtype=np.float32)

def normalize_to_256(img):
    h, w = img.shape[:2]
    if h > 256:
        top = (h - 256) // 2
        img = img[top:top+256, ...]
    if w > 256:
        left = (w - 256) // 2
        img = img[:, left:left+256, ...]
    
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

def process_dataset(label_file, output_file):
    if not os.path.exists(label_file):
        print(f"[SKIP] {label_file} tidak ditemukan.")
        return

    print(f"[PROCESS] {label_file} -> {output_file}...")
    pairs = []
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rel_path, label_str = line.split()
            pairs.append((rel_path, int(label_str)))

    X_list, y_list = [], []
    for i, (rel_path, label) in enumerate(pairs):
        if not os.path.exists(rel_path): continue
        try:
            img = iio.imread(rel_path)
            img = normalize_to_256(img)

            # 1. Warna
            feat_color = np.zeros(6, dtype=np.float32)
            if img.ndim == 3:
                feat_color = extract_color_features(img)
                img_gray = img.mean(axis=2)
            else:
                img_gray = img
            
            # 2. Wavelet
            feat_wavelet = extract_wavelet_features(img_gray)
            
            # 3. Gabung
            full = np.concatenate([feat_wavelet, feat_color])
            X_list.append(full)
            y_list.append(label)
        except Exception as e:
            print(f"Error {rel_path}: {e}")

        if i % 500 == 0: print(f"  Processed {i}/{len(pairs)}")

    if X_list:
        np.savez(output_file, X=np.vstack(X_list), y=np.array(y_list))
        print(f"Saved {output_file}. Shape: {np.vstack(X_list).shape}")

if __name__ == "__main__":
    for key, (lbl, out) in DATASETS.items():
        process_dataset(lbl, out)