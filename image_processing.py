# image_processing.py

import cv2
import numpy as np
from sklearn.decomposition import PCA
from skimage import color

def background_subtraction_luma(np_img: np.ndarray, blur_sigma: float = 25) -> np.ndarray:
    """
    Estima el fondo basándose en la LUMINANCIA (grises) y lo resta proporcionalmente
    a cada canal RGB para evitar “tintes turquesa” en las grietas.
    """
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ksize = int(2 * round(blur_sigma) + 1)
    ksize = max(ksize, 3)
    bg_gray = cv2.GaussianBlur(gray, (ksize, ksize), blur_sigma)
    sub_luma = gray - bg_gray
    minv, maxv = float(sub_luma.min()), float(sub_luma.max())
    if maxv - minv < 1e-6:
        return np_img.copy()
    sub_norm = (sub_luma - minv) / (maxv - minv)
    H, W, *_ = np_img.shape
    sub_rgb = np.zeros_like(np_img, dtype=np.uint8)
    for c in range(3):
        channel = np_img[..., c].astype(np.float32)
        corregido = channel * sub_norm
        sub_rgb[..., c] = np.clip(corregido, 0, 255).astype(np.uint8)
    return sub_rgb

def dstretch_lab(np_img: np.ndarray, low_pct: float = 1, high_pct: float = 99) -> np.ndarray:
    """
    Versión simplificada de DStretch usando espacio CIELAB.
    """
    lab = color.rgb2lab(np_img / 255.0)
    H, W, _ = lab.shape
    flat = lab.reshape(-1, 3).astype(np.float32)
    flat[:, 0] /= 100.0
    flat[:, 1] = (flat[:, 1] + 128.0) / 255.0
    flat[:, 2] = (flat[:, 2] + 128.0) / 255.0
    pca = PCA(n_components=3)
    scores = pca.fit_transform(flat)
    inv_loads = np.linalg.inv(pca.components_.T)
    stretched = np.zeros_like(scores, dtype=np.float32)
    for i in range(3):
        comp = scores[:, i]
        lo = np.percentile(comp, low_pct)
        hi = np.percentile(comp, high_pct)
        if hi <= lo:
            hi = lo + 1e-4
        norm = np.clip((comp - lo) / (hi - lo), 0, 1)
        stretched[:, i] = norm
    recon = stretched @ inv_loads
    recon[:, 0] = recon[:, 0] * 100.0
    recon[:, 1] = recon[:, 1] * 255.0 - 128.0
    recon[:, 2] = recon[:, 2] * 255.0 - 128.0
    lab_new = recon.reshape(H, W, 3).astype(np.float32)
    rgb_new = color.lab2rgb(lab_new.astype(np.float64))
    rgb_new = np.clip(rgb_new, 0, 1)
    return (rgb_new * 255).astype(np.uint8)

def apply_clahe_rgb(np_img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE al canal de luminancia y vuelve a RGB.
    """
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    rgb_clahe = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return rgb_clahe

def gamma_correction(np_img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Aplica corrección gamma en RGB.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255.0 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(np_img, table)

def enhance_rock_art(np_img: np.ndarray,
                     blur_sigma: float = 25,
                     low_pct: float = 1, high_pct: float = 99,
                     clahe_clip: float = 2.0, clahe_grid: tuple = (8, 8),
                     gamma: float = 1.0) -> np.ndarray:
    """
    Pipeline completo para realzar pinturas rupestres:
      1) Background subtraction sólo en luminancia 
      2) DStretch simplificado en CIELAB
      3) CLAHE en el resultado
      4) Corrección gamma
    """
    sub = background_subtraction_luma(np_img, blur_sigma=blur_sigma)
    dst = dstretch_lab(sub, low_pct=low_pct, high_pct=high_pct)
    clahe = apply_clahe_rgb(dst, clip_limit=clahe_clip, tile_grid_size=clahe_grid)
    final = gamma_correction(clahe, gamma=gamma)
    return final
