# ct2fea/io/file_io.py
from pathlib import Path
import tifffile
import numpy as np
import json
import logging
from typing import Tuple, Dict, List, Iterator, Optional, Union, BinaryIO
from ...config import Config


def load_ct_stack(
    input_folder: Path, load_full: bool = True
) -> Tuple[np.ndarray, List[Path]]:
    """Load CT stack from TIFF files

    Args:
        input_folder: Folder containing TIFF files
        load_full: If True, load entire stack into memory; if False, return metadata only

    Returns:
        Tuple of (CT volume array, list of file paths)
    """
    tiff_files = sorted(input_folder.glob("*.tif*"))
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_folder}")

    if load_full:
        ct_array = np.stack([tifffile.imread(f) for f in tiff_files])
        return ct_array.astype(np.float32), tiff_files
    else:
        # Just read the first slice to get dimensions
        first_slice = tifffile.imread(tiff_files[0]).astype(np.float32)
        # Create empty array with correct dimensions
        shape = (len(tiff_files),) + first_slice.shape
        return np.zeros(shape, dtype=np.float32), tiff_files


def create_ct_iterator(
    tiff_files: List[Path], chunk_size: int = 10
) -> Iterator[np.ndarray]:
    """Create iterator for loading CT data in chunks

    Args:
        tiff_files: List of TIFF file paths
        chunk_size: Number of slices to load at once

    Yields:
        Chunks of CT data as numpy arrays
    """
    logger = logging.getLogger(__name__)
    total_files = len(tiff_files)

    for i in range(0, total_files, chunk_size):
        end_idx = min(i + chunk_size, total_files)
        logger.info(f"Loading CT chunk {i}-{end_idx - 1} of {total_files}")

        chunk_files = tiff_files[i:end_idx]
        chunk_data = np.stack([tifffile.imread(f) for f in chunk_files])

        yield chunk_data.astype(np.float32)


def get_ct_metadata(input_folder: Path) -> Dict[str, any]:
    """Get metadata about CT stack without loading full data

    Args:
        input_folder: Folder containing TIFF files

    Returns:
        Dictionary with metadata (dimensions, file count, etc.)
    """
    tiff_files = sorted(input_folder.glob("*.tif*"))
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_folder}")

    # Read first slice to get dimensions
    first_slice = tifffile.imread(tiff_files[0])

    return {
        "n_slices": len(tiff_files),
        "width": first_slice.shape[1],
        "height": first_slice.shape[0],
        "dtype": first_slice.dtype,
        "file_list": [str(f.name) for f in tiff_files],
        "total_size_mb": len(tiff_files) * first_slice.nbytes / (1024 * 1024),
    }


def save_processed_ct(output_dir: Path, ct_array: np.ndarray, config: Config) -> None:
    """Save processed CT data to TIFF file

    Args:
        output_dir: Output directory
        ct_array: CT volume data
        config: Configuration object
    """
    output_path = output_dir / "processed_ct_stack.tif"
    tifffile.imwrite(output_path, (ct_array * 65535).astype(np.uint16))

    meta = {
        "voxel_size_um": config.voxel_size_um,
        "coarsen_factor": config.coarsen_factor,
        "clip_percentiles": list(config.clip_percentiles),
        "dimensions": list(ct_array.shape),
    }
    with open(output_dir / "processing_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)


def save_ct_chunks(
    output_dir: Path, chunks: Iterator[np.ndarray], config: Config
) -> Path:
    """Save CT data in chunks to avoid memory issues

    Args:
        output_dir: Output directory
        chunks: Iterator providing CT data chunks
        config: Configuration object

    Returns:
        Path to saved file
    """
    logger = logging.getLogger(__name__)
    output_path = output_dir / "processed_ct_stack.tif"

    # Get first chunk to determine dimensions
    try:
        first_chunk = next(chunks)
        chunk_shape = first_chunk.shape

        # Write first chunk and prepare for appending
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            tif.write((first_chunk * 65535).astype(np.uint16))

            # Write remaining chunks
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Saving CT chunk {i}")
                tif.write((chunk * 65535).astype(np.uint16))

        logger.info(f"Saved processed CT data to {output_path}")
        return output_path

    except StopIteration:
        logger.warning("No CT chunks to save")
        return None
