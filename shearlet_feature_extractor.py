import numpy as np
from FFST import shearletTransformSpect

def extract_shearlet_features(img: np.ndarray) -> np.ndarray:
    # 
    # img: array 2D float32, shape (H, W), misal (256, 256)
    # return: 1D array fitur, shape (3 * jumlah_band)
    # 
    ST, Psi = shearletTransformSpect(img)

    feats = []
    for k in range(ST.shape[2]):
        band = ST[:, :, k]
        mag  = np.abs(band)

        energy = np.sum(mag**2)
        mean   = np.mean(mag)
        var    = np.var(mag)

        feats.extend([energy, mean, var])

    return np.array(feats, dtype=np.float32)
