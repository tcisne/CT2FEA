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

    # Parallel Processing
    use_parallel: bool = True
    n_jobs: Optional[int] = None  # None means use all available cores
    chunk_size: Optional[int] = None  # None means auto-determine

    # Denoising
    denoise_method: Optional[str] = field(
        default=None, metadata={"choices": ["gaussian", "nlm", "tv"]}
    )
    denoise_strength: float = 1.0

    # Segmentation
    pore_threshold: float = 0.2

    # Material Properties
    material_model: str = field(
        default="linear", metadata={"choices": ["linear", "plasticity", "hyperelastic"]}
    )
    material_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "min_density": 1.0,
            "max_density": 2.0,
            "E_coeff": (10000, 2.0),
            "yield_stress": 100.0,
            "hardening_coeff": 0.1,
        }
    )

    # GPU Acceleration
    use_gpu: bool = True
    gpu_params: Dict[str, Any] = field(
        default_factory=lambda: {"block_size": 256, "use_opencl_fallback": True}
    )

    # Visualization
    save_visualization: bool = True
    visualization_dpi: int = 300

    @classmethod
    def from_gui(cls):
        """Create Config instance from GUI inputs"""
        from .io.gui import get_gui_inputs

        return get_gui_inputs()

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
