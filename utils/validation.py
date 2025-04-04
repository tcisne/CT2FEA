# ct2fea/utils/validation.py
import numpy as np
import pyvista as pv
from typing import Dict, Any, Tuple
from pathlib import Path
import logging
from ..utils.errors import (
    CTDataError,
    SegmentationError,
    MeshingError,
    MaterialError,
    ConfigError,
)


def validate_ct_data(data: np.ndarray, metadata: Dict[str, Any]) -> None:
    """Validate CT data and metadata

    Args:
        data: CT volume data
        metadata: Associated metadata

    Raises:
        CTDataError: If validation fails
    """
    logger = logging.getLogger(__name__)

    # Check data type and dimensions
    if not isinstance(data, np.ndarray):
        raise CTDataError("CT data must be a numpy array")

    if data.ndim != 3:
        raise CTDataError("CT data must be 3-dimensional", f"Got shape: {data.shape}")

    # Check for NaN or infinite values
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise CTDataError("CT data contains NaN or infinite values")

    # Validate value range
    if data.min() < 0:
        logger.warning("CT data contains negative values")

    # Log data statistics
    logger.info(
        f"CT data statistics: shape={data.shape}, "
        f"range=[{data.min():.2f}, {data.max():.2f}], "
        f"mean={data.mean():.2f}, std={data.std():.2f}"
    )


def validate_segmentation(
    bone_mask: np.ndarray, pore_mask: np.ndarray, min_volume_fraction: float = 0.001
) -> None:
    """Validate segmentation results

    Args:
        bone_mask: Binary bone mask
        pore_mask: Binary pore mask
        min_volume_fraction: Minimum acceptable volume fraction

    Raises:
        SegmentationError: If validation fails
    """
    # Check mask types
    if bone_mask.dtype != np.bool_ or pore_mask.dtype != np.bool_:
        raise SegmentationError("Segmentation masks must be boolean arrays")

    # Check for overlapping regions
    if np.any(np.logical_and(bone_mask, pore_mask)):
        raise SegmentationError("Bone and pore masks overlap")

    # Check minimum volume fractions
    bone_fraction = np.mean(bone_mask)
    pore_fraction = np.mean(pore_mask)

    if bone_fraction < min_volume_fraction:
        raise SegmentationError(
            "Insufficient bone volume detected", f"Bone fraction: {bone_fraction:.4f}"
        )

    logging.getLogger(__name__).info(
        f"Segmentation fractions - Bone: {bone_fraction:.4f}, Pore: {pore_fraction:.4f}"
    )


def validate_mesh(mesh: pv.UnstructuredGrid, min_quality: float = 0.1) -> None:
    """Validate mesh quality

    Args:
        mesh: PyVista mesh
        min_quality: Minimum acceptable element quality

    Raises:
        MeshingError: If validation fails
    """
    logger = logging.getLogger(__name__)

    if mesh.n_cells == 0:
        raise MeshingError("Empty mesh generated")

    if mesh.n_points < 8:
        raise MeshingError("Insufficient mesh nodes")

    # Compute quality metrics
    quality = mesh.compute_cell_quality()
    min_qual = quality["CellQuality"].min()

    if min_qual < min_quality:
        raise MeshingError(
            f"Mesh contains poor quality elements (min quality: {min_qual:.3f})",
            f"Number of poor elements: {np.sum(quality['CellQuality'] < min_quality)}",
        )

    # Check for inverted elements
    if min_qual <= 0:
        raise MeshingError("Mesh contains inverted elements")

    logger.info(
        f"Mesh validation passed - Elements: {mesh.n_cells}, "
        f"Nodes: {mesh.n_points}, Min quality: {min_qual:.3f}"
    )


def validate_material_properties(
    properties: Dict[str, np.ndarray], material_model: str
) -> None:
    """Validate material property calculations

    Args:
        properties: Dictionary of material properties
        material_model: Material model type

    Raises:
        MaterialError: If validation fails
    """
    required_props = {
        "linear": ["density", "youngs_modulus", "poissons_ratio"],
        "plasticity": [
            "density",
            "youngs_modulus",
            "poissons_ratio",
            "yield_stress",
            "hardening_coeff",
        ],
        "hyperelastic": ["density", "c10", "d1"],
    }

    # Check required properties
    if material_model not in required_props:
        raise MaterialError(f"Unknown material model: {material_model}")

    missing = [p for p in required_props[material_model] if p not in properties]
    if missing:
        raise MaterialError(
            f"Missing required properties for {material_model} model",
            f"Missing: {', '.join(missing)}",
        )

    # Validate property ranges
    for name, prop in properties.items():
        if np.any(np.isnan(prop)) or np.any(np.isinf(prop)):
            raise MaterialError(f"Property '{name}' contains invalid values")
        if name != "poissons_ratio" and np.any(prop <= 0):
            raise MaterialError(f"Property '{name}' contains non-positive values")
        if name == "poissons_ratio" and np.any(np.abs(prop) >= 0.5):
            raise MaterialError("Invalid Poisson's ratio values detected")


def validate_output_path(path: Path) -> None:
    """Validate output path

    Args:
        path: Output directory path

    Raises:
        ConfigError: If validation fails
    """
    try:
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # Check write permissions
        test_file = path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise ConfigError(
                "Output directory is not writable", f"Path: {path}, Error: {str(e)}"
            )

    except Exception as e:
        raise ConfigError(
            "Failed to validate output directory", f"Path: {path}, Error: {str(e)}"
        )
