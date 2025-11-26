import numpy as np
import imageio.v2 as iio
from FFST import shearletTransformSpect, inverseShearletTransformSpect

def load_gray_image(path):
    # baca gambar dan paksa jadi grayscale float32
    img = iio.imread(path)
    if img.ndim == 3:
        # kalau RGB, ambil rata-rata channel
        img = img.mean(axis=2)
    img = img.astype(np.float32)

    # optional: normalisasi biar stabil
    img = img - img.mean()
    img = img / (img.std() + 1e-8)
    return img

def main():
    # TODO: ganti ini jadi path ke satu gambar uji (boleh apa saja dulu)
    img_path = "test.png"

    img = load_gray_image(img_path)
    print("Image shape:", img.shape)

    # --- shearlet transform ---
    # Python port-nya dipakai seperti ini:
    # ST = koefisien shearlet (M x N x K)
    # Psi = spektra shearlet (M x N x K)
    ST, Psi = shearletTransformSpect(img)

    print("ST shape (coeffs):", ST.shape)
    print("Psi shape (spectra):", Psi.shape)

    # --- inverse transform (cek bisa balik lagi) ---
    recon = inverseShearletTransformSpect(ST, Psi)

    # hitung error rekonstruksi
    err = np.abs(recon - img).max()
    print("Max reconstruction error:", err)

        # =============================
    # 4️⃣ EKSTRAKSI FITUR DARI KOEFISIEN SHEARLET
    # =============================

    features = []

    # ST shape: (H, W, K) → kita ringkas per channel K
    for k in range(ST.shape[2]):
        band = ST[:, :, k]
        mag  = np.abs(band)

        energy = np.sum(mag**2)   # energi band
        mean   = np.mean(mag)     # rata-rata
        var    = np.var(mag)      # variansi

        features.extend([energy, mean, var])

    features = np.array(features, dtype=np.float32)

    print("Panjang vektor fitur:", features.shape[0])
    print("10 fitur pertama:", features[:10])


if __name__ == "__main__":
    main()
