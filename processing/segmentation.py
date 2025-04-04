# ct2fea/processing/segmentation.py
import numpy as np
from skimage.filters import (
    threshold_otsu,
    threshold_local,
    threshold_triangle,
    threshold_yen,
)
from skimage.morphology import binary_closing, binary_opening, remove_small_objects
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes
from skimage.measure import regionprops
from skimage.exposure import histogram
from typing import Tuple, Optional, Dict, Any, List
import logging
from ..config import Config
from ..utils.parallel import parallel_map_slices, parallel_map_blocks

logger = logging.getLogger(__name__)


def analyze_histogram(ct_volume: np.ndarray) -> Dict[str, Any]:
    """Analyze histogram to determine optimal segmentation parameters

    Args:
        ct_volume: Normalized CT volume data

    Returns:
        Dictionary with analysis results and recommended parameters
    """
    # Calculate histogram
    hist, bin_centers = histogram(ct_volume)

    # Find peaks in histogram (simplified approach)
    # In a real implementation, you would use more sophisticated peak detection
    smoothed_hist = np.convolve(hist, np.ones(5) / 5, mode="same")
    peaks = []
    for i in range(1, len(smoothed_hist) - 1):
        if (
            smoothed_hist[i] > smoothed_hist[i - 1]
            and smoothed_hist[i] > smoothed_hist[i + 1]
        ):
            if smoothed_hist[i] > np.mean(smoothed_hist):  # Only significant peaks
                peaks.append((bin_centers[i], smoothed_hist[i]))

    # Sort peaks by height
    peaks.sort(key=lambda x: x[1], reverse=True)

    # Calculate various thresholds for comparison
    otsu_thresh = threshold_otsu(ct_volume)
    try:
        triangle_thresh = threshold_triangle(ct_volume)
    except:
        triangle_thresh = otsu_thresh
    try:
        yen_thresh = threshold_yen(ct_volume)
    except:
        yen_thresh = otsu_thresh

    # Determine if adaptive thresholding is needed
    # If histogram has multiple strong peaks, adaptive thresholding may be better
    use_adaptive = len(peaks) > 2

    # Determine optimal block size for adaptive thresholding
    # Larger block sizes for more uniform data, smaller for detailed data
    intensity_variance = np.var(ct_volume)
    if intensity_variance > 0.05:  # High variance suggests detailed data
        block_size = 51  # Smaller block size
    else:
        block_size = 101  # Larger block size

    # Determine optimal minimum object size based on volume dimensions
    volume_size = np.prod(ct_volume.shape)
    min_bone_size = max(50, int(volume_size * 0.0001))  # 0.01% of volume size
    min_pore_size = max(20, int(volume_size * 0.00005))  # 0.005% of volume size

    return {
        "peaks": peaks,
        "thresholds": {
            "otsu": float(otsu_thresh),
            "triangle": float(triangle_thresh),
            "yen": float(yen_thresh),
        },
        "recommended_params": {
            "use_adaptive_threshold": use_adaptive,
            "adaptive_block_size": block_size,
            "min_bone_size": min_bone_size,
            "min_pore_size": min_pore_size,
            "pore_threshold": max(0.1, min(0.3, float(triangle_thresh))),
        },
        "intensity_variance": float(intensity_variance),
    }


def segment_volume(
    ct_volume: np.ndarray, config: Config, adaptive_params: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment CT volume into bone and pore regions with parallel processing

    Args:
        ct_volume: Normalized CT volume data
        config: Configuration object
        adaptive_params: Whether to use adaptive parameter selection

    Returns:
        Tuple of (bone_mask, pore_mask) as binary arrays
    """
    # Analyze data and determine optimal parameters if adaptive mode is enabled
    if adaptive_params:
        logger.info("Analyzing CT data for adaptive parameter selection")
        analysis = analyze_histogram(ct_volume)
        logger.info(
            f"Histogram analysis complete - detected {len(analysis['peaks'])} peaks"
        )
        logger.info(f"Recommended parameters: {analysis['recommended_params']}")

        # Update segmentation parameters with recommended values
        segmentation_params = (
            config.segmentation_params.copy()
            if hasattr(config, "segmentation_params")
            else {}
        )
        segmentation_params.update(analysis["recommended_params"])
    else:
        # Use parameters from config
        segmentation_params = (
            config.segmentation_params if hasattr(config, "segmentation_params") else {}
        )
    # Bone segmentation with adaptive thresholding
    if segmentation_params.get("use_adaptive_threshold", False):
        logger.info(
            f"Using adaptive thresholding with block size {segmentation_params.get('adaptive_block_size', 99)}"
        )

        def threshold_slice(slice_data: np.ndarray) -> np.ndarray:
            block_size = segmentation_params.get("adaptive_block_size", 99)
            thresh = threshold_local(slice_data, block_size, offset=0)
            return slice_data > thresh

        if config.use_parallel and ct_volume.shape[0] > 50:
            bone_mask = parallel_map_slices(
                threshold_slice, ct_volume, n_jobs=config.n_jobs
            )
        else:
            block_size = segmentation_params.get("adaptive_block_size", 99)
            bone_thresh = threshold_local(ct_volume, block_size, offset=0)
            bone_mask = ct_volume > bone_thresh
    else:
        logger.info("Using Otsu thresholding for bone segmentation")
        bone_thresh = threshold_otsu(ct_volume)
        bone_mask = ct_volume > bone_thresh

    # Clean up bone mask
    min_bone_size = segmentation_params.get("min_bone_size", 100)
    logger.info(f"Removing small bone objects (min size: {min_bone_size})")
    bone_mask = remove_small_objects(bone_mask, min_size=min_bone_size)
    bone_mask = binary_closing(bone_mask)
    bone_mask = binary_fill_holes(bone_mask)

    # Calculate bone statistics
    bone_volume = np.sum(bone_mask)
    bone_fraction = bone_volume / bone_mask.size
    logger.info(f"Bone segmentation complete - volume fraction: {bone_fraction:.3f}")

    # Pore segmentation with parallel processing
    pore_thresh = segmentation_params.get("pore_threshold", 0.2)
    logger.info(f"Using pore threshold: {pore_thresh}")

    def process_chunk(chunk: np.ndarray) -> np.ndarray:
        air = chunk < pore_thresh
        # Remove small pores within chunk
        cleaned = remove_small_objects(
            air, min_size=segmentation_params.get("min_pore_size", 20)
        )
        return cleaned

    if config.use_parallel and ct_volume.shape[0] > 50:
        air_mask = parallel_map_blocks(process_chunk, ct_volume, n_jobs=config.n_jobs)
    else:
        air_mask = process_chunk(ct_volume)

    # Connected component analysis for pores
    structure = generate_binary_structure(3, 1)
    labels, _ = label(air_mask, structure=structure)

    # Remove external air by identifying edge-connected components
    edge_labels = np.unique(
        np.concatenate(
            [
                labels[0],
                labels[-1],
                labels[:, 0],
                labels[:, -1],
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            ]
        )
    )

    # Create pore mask (internal air only)
    pore_mask = ~np.isin(labels, edge_labels)

    # Clean up pore mask
    min_pore_size = segmentation_params.get("min_pore_size", 20)
    logger.info(f"Removing small pore objects (min size: {min_pore_size})")
    pore_mask = remove_small_objects(pore_mask, min_size=min_pore_size)
    pore_mask = binary_opening(pore_mask)

    # Ensure no overlap between bone and pores
    pore_mask[bone_mask] = False

    # Log segmentation statistics
    bone_fraction = np.sum(bone_mask) / bone_mask.size
    pore_fraction = np.sum(pore_mask) / pore_mask.size
    logger.info(
        f"Segmentation complete - Bone fraction: {bone_fraction:.3f}, "
        f"Pore fraction: {pore_fraction:.3f}"
    )

    return bone_mask, pore_mask


def validate_segmentation(
    bone_mask: np.ndarray, pore_mask: np.ndarray
) -> Dict[str, Any]:
    """Validate segmentation results and return quality metrics

    Args:
        bone_mask: Binary bone mask
        pore_mask: Binary pore mask

    Returns:
        Dictionary with validation metrics

    Raises:
        ValueError: If validation fails
    """
    metrics = {}

    # Check for overlap
    overlap = np.logical_and(bone_mask, pore_mask)
    overlap_count = np.sum(overlap)
    if overlap_count > 0:
        raise ValueError(
            f"Overlap detected between bone and pore masks ({overlap_count} voxels)"
        )

    # Check for empty bone mask
    bone_count = np.sum(bone_mask)
    if bone_count == 0:
        raise ValueError("No bone tissue detected in segmentation")

    # Check data types
    if bone_mask.dtype != np.bool_ or pore_mask.dtype != np.bool_:
        raise ValueError("Segmentation masks must be boolean arrays")

    # Calculate segmentation metrics
    total_voxels = bone_mask.size
    bone_fraction = bone_count / total_voxels
    pore_count = np.sum(pore_mask)
    pore_fraction = pore_count / total_voxels
    background_fraction = 1.0 - bone_fraction - pore_fraction

    # Calculate connectivity metrics for bone
    structure = generate_binary_structure(3, 1)
    bone_labels, bone_components = label(bone_mask, structure=structure)

    # Get region properties
    if bone_components > 0:
        bone_regions = regionprops(bone_labels)
        bone_sizes = [region.area for region in bone_regions]
        largest_bone_fraction = max(bone_sizes) / bone_count if bone_count > 0 else 0
    else:
        bone_sizes = []
        largest_bone_fraction = 0

    metrics = {
        "bone_fraction": float(bone_fraction),
        "pore_fraction": float(pore_fraction),
        "background_fraction": float(background_fraction),
        "bone_components": int(bone_components),
        "largest_bone_fraction": float(largest_bone_fraction),
        "bone_voxels": int(bone_count),
        "pore_voxels": int(pore_count),
    }

    logger.info(f"Segmentation validation: {metrics}")
    return metrics
