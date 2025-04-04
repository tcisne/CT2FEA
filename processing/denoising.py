import numpy as np
from typing import Optional, Dict, Any
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
import logging
from ..config import Config


def denoise_volume(
    volume: np.ndarray, method: Optional[str], config: Config
) -> np.ndarray:
    """Apply selected denoising method with automatic parameter estimation

    Args:
        volume: Input CT volume
        method: Denoising method name
        config: Configuration object

    Returns:
        Denoised volume
    """
    logger = logging.getLogger(__name__)

    if method is None:
        return volume

    # Estimate noise level for parameter tuning
    sigma_est = estimate_sigma(volume)
    logger.info(f"Estimated noise level (sigma): {sigma_est:.3f}")

    # Get method-specific parameters
    params = config.denoise_params.get(method, {})

    if method == "gaussian":
        return _apply_gaussian(volume, sigma_est, params)
    elif method == "nlm":
        return _apply_nlm(volume, sigma_est, params)
    elif method == "tv":
        return _apply_tv(volume, sigma_est, params)
    elif method == "median":
        return _apply_median(volume, params)
    elif method == "hybrid":
        return _apply_hybrid(volume, sigma_est, params)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def _apply_gaussian(
    volume: np.ndarray, sigma_est: float, params: Dict[str, Any]
) -> np.ndarray:
    """Apply Gaussian filtering with adaptive sigma"""
    sigma = params.get("sigma_factor", 1.0) * sigma_est
    return gaussian_filter(volume, sigma=sigma)


def _apply_nlm(
    volume: np.ndarray, sigma_est: float, params: Dict[str, Any]
) -> np.ndarray:
    """Apply Non-local Means denoising with optimized parameters"""
    patch_size = params.get("patch_size", 5)
    patch_distance = params.get("patch_distance", 6)
    h = params.get("h_factor", 1.15) * sigma_est

    return denoise_nl_means(
        volume,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        fast_mode=params.get("fast_mode", True),
        sigma=sigma_est,
    )


def _apply_tv(
    volume: np.ndarray, sigma_est: float, params: Dict[str, Any]
) -> np.ndarray:
    """Apply Total Variation denoising"""
    weight = params.get("weight_factor", 0.1) * sigma_est
    return denoise_tv_chambolle(
        volume,
        weight=weight,
        eps=params.get("eps", 1e-4),
        n_iter_max=params.get("max_iter", 200),
    )


def _apply_median(volume: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply 3D median filtering"""
    size = params.get("kernel_size", 3)
    return median_filter(volume, size=size)


def _apply_hybrid(
    volume: np.ndarray, sigma_est: float, params: Dict[str, Any]
) -> np.ndarray:
    """Apply hybrid denoising (combine multiple methods)

    First applies edge-preserving TV denoising, then NLM for detail preservation
    """
    # First stage: TV denoising
    tv_params = params.get("tv_params", {"weight_factor": 0.1})
    volume_tv = _apply_tv(volume, sigma_est, tv_params)

    # Second stage: NLM refinement
    nlm_params = params.get("nlm_params", {"h_factor": 0.8, "fast_mode": True})
    volume_final = _apply_nlm(volume_tv, sigma_est * 0.5, nlm_params)

    return volume_final


def evaluate_denoising(original: np.ndarray, denoised: np.ndarray) -> Dict[str, float]:
    """Evaluate denoising quality using multiple metrics

    Args:
        original: Original volume
        denoised: Denoised volume

    Returns:
        Dictionary of quality metrics
    """
    # Normalize to [0,1] for PSNR calculation
    orig_norm = (original - original.min()) / (original.max() - original.min())
    den_norm = (denoised - denoised.min()) / (denoised.max() - denoised.min())

    psnr = peak_signal_noise_ratio(orig_norm, den_norm)
    noise_reduction = np.std(original) / np.std(denoised)

    return {
        "psnr": float(psnr),
        "noise_reduction_factor": float(noise_reduction),
        "mean_difference": float(np.mean(np.abs(original - denoised))),
        "max_difference": float(np.max(np.abs(original - denoised))),
    }
