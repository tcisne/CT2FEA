# ct2fea/processing/segmentation.py
import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import binary_closing, binary_opening, remove_small_objects
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes
from typing import Tuple, Optional
import logging
from ..config import Config
from ..utils.parallel import parallel_map_slices, parallel_map_blocks

logger = logging.getLogger(__name__)


def segment_volume(
    ct_volume: np.ndarray, config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment CT volume into bone and pore regions with parallel processing

    Args:
        ct_volume: Normalized CT volume data
        config: Configuration object

    Returns:
        Tuple of (bone_mask, pore_mask) as binary arrays
    """
    # Bone segmentation with adaptive thresholding
    if config.segmentation_params.get("use_adaptive_threshold", False):

        def threshold_slice(slice_data: np.ndarray) -> np.ndarray:
            block_size = config.segmentation_params.get("adaptive_block_size", 99)
            thresh = threshold_local(slice_data, block_size, offset=0)
            return slice_data > thresh

        if config.use_parallel and ct_volume.shape[0] > 50:
            bone_mask = parallel_map_slices(
                threshold_slice, ct_volume, n_jobs=config.n_jobs
            )
        else:
            bone_thresh = threshold_local(ct_volume, block_size, offset=0)
            bone_mask = ct_volume > bone_thresh
    else:
        bone_thresh = threshold_otsu(ct_volume)
        bone_mask = ct_volume > bone_thresh

    # Clean up bone mask
    min_bone_size = config.segmentation_params.get("min_bone_size", 100)
    bone_mask = remove_small_objects(bone_mask, min_size=min_bone_size)
    bone_mask = binary_closing(bone_mask)
    bone_mask = binary_fill_holes(bone_mask)

    # Pore segmentation with parallel processing
    pore_thresh = config.segmentation_params.get("pore_threshold", 0.2)

    def process_chunk(chunk: np.ndarray) -> np.ndarray:
        air = chunk < pore_thresh
        # Remove small pores within chunk
        cleaned = remove_small_objects(
            air, min_size=config.segmentation_params.get("min_pore_size", 20)
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
    min_pore_size = config.segmentation_params.get("min_pore_size", 20)
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


def validate_segmentation(bone_mask: np.ndarray, pore_mask: np.ndarray) -> None:
    """Validate segmentation results

    Args:
        bone_mask: Binary bone mask
        pore_mask: Binary pore mask

    Raises:
        ValueError: If validation fails
    """
    if np.any(np.logical_and(bone_mask, pore_mask)):
        raise ValueError("Overlap detected between bone and pore masks")

    if np.sum(bone_mask) == 0:
        raise ValueError("No bone tissue detected in segmentation")

    if bone_mask.dtype != np.bool_ or pore_mask.dtype != np.bool_:
        raise ValueError("Segmentation masks must be boolean arrays")
