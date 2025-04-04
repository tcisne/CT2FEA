# ct2fea/io/file_io.py
from pathlib import Path
import tifffile
import numpy as np
import json
from typing import Tuple, Dict
from ...config import Config


def load_ct_stack(input_folder: Path) -> Tuple[np.ndarray, list]:
    tiff_files = sorted(input_folder.glob("*.tif*"))
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_folder}")

    ct_array = np.stack([tifffile.imread(f) for f in tiff_files])
    return ct_array.astype(np.float32), tiff_files


def save_processed_ct(output_dir: Path, ct_array: np.ndarray, config: Config) -> None:
    output_path = output_dir / "processed_ct_stack.tif"
    tifffile.imwrite(output_path, (ct_array * 65535).astype(np.uint16))

    meta = {
        "voxel_size_um": config.voxel_size_um,
        "coarsen_factor": config.coarsen_factor,
        "clip_percentiles": list(config.clip_percentiles),
    }
    with open(output_dir / "processing_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
