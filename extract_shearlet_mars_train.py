import os
import numpy as np
import imageio.v2 as iio
from FFST import shearletTransformSpect


# ==========================
# 1. PARAMETER DATASET
# ==========================
# Kalau file .txt & folder 'calibrated' ada di folder yang sama dengan script ini,
# kamu nggak perlu ubah apa-apa.
DATA_ROOT   = "."   # root folder dataset (di sini sama dengan current dir)
LABEL_FILE  = os.path.join(DATA_ROOT, "train-calibrated-shuffled.txt")
OUTPUT_NPZ  = os.path.join(DATA_ROOT, "shearlet_train.npz")


# ==========================
# 2. NORMALISASI UKURAN -> 256x256
# ==========================
def normalize_to_256(img: np.ndarray) -> np.ndarray:
    """
    Pastikan citra jadi 256x256:
      - Jika lebih besar: crop tengah
      - Jika lebih kecil: pad pinggir dengan nilai piksel tepi (mode='edge')
    """
    h, w = img.shape

    # 1) Crop tengah kalau lebih besar dari 256
    if h > 256:
        top = (h - 256) // 2
        img = img[top:top+256, :]
    if w > 256:
        left = (w - 256) // 2
        img = img[:, left:left+256]

    # update ukuran
    h, w = img.shape

    # 2) Pad kalau lebih kecil dari 256
    pad_top    = max((256 - h) // 2, 0)
    pad_bottom = 256 - h - pad_top
    pad_left   = max((256 - w) // 2, 0)
    pad_right  = 256 - w - pad_left

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="edge"
        )

    return img


# ==========================
# 3. EKSTRAKSI FITUR SHEARLET UNTUK 1 CITRA
# ==========================
def extract_shearlet_features(img: np.ndarray) -> np.ndarray:
    """
    img: array 2D float32, shape (H, W)
    return: 1D array fitur, shape (3 * jumlah_band)
    """
    ST, Psi = shearletTransformSpect(img)  # ST shape: (H, W, K)

    feats = []
    for k in range(ST.shape[2]):
        band = ST[:, :, k]
        mag  = np.abs(band)

        energy = np.sum(mag**2)
        mean   = np.mean(mag)
        var    = np.var(mag)

        feats.extend([energy, mean, var])

    return np.array(feats, dtype=np.float32)


# ==========================
# 4. LOAD LIST GAMBAR + LABEL DARI TXT
# ==========================
def load_image_label_pairs(label_file: str):
    """
    Baca train-calibrated-shuffled.txt
    Return: list of (full_image_path, label_int)
    """
    pairs = []
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label_str = line.split()
            # rel_path biasanya "calibrated/namafile.png"
            full_path = os.path.join(DATA_ROOT, rel_path)
            pairs.append((full_path, int(label_str)))
    return pairs


# ==========================
# 5. MAIN: LOOP SEMUA TRAIN IMAGE
# ==========================
def main():
    pairs = load_image_label_pairs(LABEL_FILE)
    print("Total train samples:", len(pairs))

    X_list = []
    y_list = []

    for i, (img_path, label) in enumerate(pairs, start=1):
        if not os.path.exists(img_path):
            print(f"[WARNING] File tidak ditemukan, skip: {img_path}")
            continue

        try:
            img = iio.imread(img_path)

            # pastikan grayscale 2D
            if img.ndim == 3:
                img = img.mean(axis=2)

            img = img.astype(np.float32)

            # normalisasi ukuran -> 256x256
            img = normalize_to_256(img)

            # ekstraksi fitur shearlet
            feats = extract_shearlet_features(img)

            X_list.append(feats)
            y_list.append(label)

        except Exception as e:
            print(f"[ERROR] Gagal proses {img_path}: {e}")
            continue

        if i % 50 == 0 or i == len(pairs):
            print(f"Processed {i}/{len(pairs)}")

    if not X_list:
        print("Tidak ada sampel yang berhasil diproses.")
        return

    X = np.vstack(X_list)                 # shape: (N, n_fitur)
    y = np.array(y_list, dtype=np.int64)  # shape: (N,)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    np.savez(OUTPUT_NPZ, X=X, y=y)
    print("Saved to:", OUTPUT_NPZ)


if __name__ == "__main__":
    main()
