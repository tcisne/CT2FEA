# ct2fea/config.py
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


@dataclass
class Config:
    """Configuration for CT2FEA pipeline

    Handles all parameters needed for CT processing, meshing, and FEA export
    """

    # Input/Output
    input_folder: str
    output_dir: str

    # CT Processing
    voxel_size_um: float = 10.0
    clip_percentiles: Tuple[float, float] = field(default=(0.1, 99.9))
    coarsen_factor: int = 1

    # Processing Options
    use_parallel: bool = True
    n_jobs: Optional[int] = None  # None means use all available cores
    chunk_size: int = 10  # Number of slices to process at once

    # Denoising
    denoise_method: Optional[str] = field(
        default=None, metadata={"choices": ["gaussian", "nlm"]}
    )
    denoise_strength: float = 1.0

    # Segmentation
    pore_threshold: float = 0.2
    segmentation_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "use_adaptive_threshold": False,
            "adaptive_block_size": 99,
            "min_bone_size": 100,
            "min_pore_size": 20,
            "pore_threshold": 0.2,
        }
    )

    # Material Properties
    material_model: str = field(default="linear", metadata={"choices": ["linear"]})
    material_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "min_density": 1.0,
            "max_density": 2.0,
            "E_coeff": (10000, 2.0),
        }
    )

    # Visualization
    save_visualization: bool = True
    visualization_dpi: int = 300

    def validate(self) -> None:
        """Validate configuration parameters"""
        if not Path(self.input_folder).exists():
            raise ValueError(f"Input folder does not exist: {self.input_folder}")
        if self.voxel_size_um <= 0:
            raise ValueError(f"Invalid voxel size: {self.voxel_size_um}")
        if self.coarsen_factor < 1:
            raise ValueError(f"Invalid coarsen factor: {self.coarsen_factor}")
        if not 0 <= self.pore_threshold <= 1:
            raise ValueError(f"Invalid pore threshold: {self.pore_threshold}")
        if self.chunk_size < 1:
            raise ValueError(f"Invalid chunk size: {self.chunk_size}")
