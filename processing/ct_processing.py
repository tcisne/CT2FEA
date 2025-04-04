# ct2fea/processing/ct_processing.py
import numpy as np
from typing import Tuple, Dict
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means
from ..utils.parallel import parallel_map_slices, parallel_map_blocks
from ..config import Config


def normalize_ct_volume(
    ct_volume: np.ndarray, clip_percentiles: Tuple[float, float]
) -> Tuple[np.ndarray, Dict]:
    """Normalize CT volume with parallel processing for large volumes"""
    # Calculate global percentiles
    min_val = np.percentile(ct_volume, clip_percentiles[0])
    max_val = np.percentile(ct_volume, clip_percentiles[1])

    # Define normalization function for parallel processing
    def normalize_chunk(chunk: np.ndarray) -> np.ndarray:
        clipped = np.clip(chunk, min_val, max_val)
        return ((clipped - min_val) / (max_val - min_val)).astype(np.float32)

    # Process in parallel if volume is large enough
    if ct_volume.shape[0] > 50:  # Threshold for parallel processing
        normalized = parallel_map_blocks(normalize_chunk, ct_volume)
    else:
        normalized = normalize_chunk(ct_volume)

    return normalized, {
        "clip_min": float(min_val),
        "clip_max": float(max_val),
    }


def coarsen_volume(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return volume
    return volume[::factor, ::factor, ::factor]


def denoise_volume(volume: np.ndarray, method: str, config: Config) -> np.ndarray:
    """Apply denoising with parallel processing for large volumes"""

    if method == "nlm":
        # Non-local means is already parallel with scikit-image
        return denoise_nl_means(
            volume,
            patch_size=5,
            patch_distance=3,
            h=0.1,
            fast_mode=True,
            n_jobs=config.n_jobs if config.use_parallel else 1,
        )
    elif method == "gaussian":
        # Define gaussian filter function for parallel processing
        def apply_gaussian(chunk: np.ndarray) -> np.ndarray:
            return gaussian_filter(chunk, sigma=1)

        if config.use_parallel and volume.shape[0] > 50:
            return parallel_map_slices(apply_gaussian, volume, n_jobs=config.n_jobs)
        else:
            return apply_gaussian(volume)

    return volume
