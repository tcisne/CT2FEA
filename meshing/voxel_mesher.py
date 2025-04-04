# ct2fea/meshing/voxel_mesher.py
import pyvista as pv
import numpy as np
from typing import Dict, Tuple
import logging
from ..config import Config
from ..utils.validation import validate_mesh


def create_voxel_mesh(
    bone_mask: np.ndarray, pore_mask: np.ndarray, config: Config
) -> Tuple[pv.UnstructuredGrid, Dict]:
    """Create hexahedral mesh from segmented volume data

    Args:
        bone_mask: Binary mask of bone regions
        pore_mask: Binary mask of pore regions
        config: Configuration object

    Returns:
        Tuple of (mesh, material_info)
    """
    logger = logging.getLogger(__name__)

    # Create base grid
    grid = pv.UniformGrid()
    grid.dimensions = np.array(bone_mask.shape) + 1
    grid.spacing = (config.voxel_size_um / 1000.0,) * 3  # Convert to mm

    # Create material array and mapping
    materials, material_info = _create_material_mapping(bone_mask, pore_mask)
    grid.cell_data["Material"] = materials

    # Extract active cells (either bone or pores)
    threshold = grid.threshold(0.5, scalars="Material")

    # Ensure mesh quality
    validate_mesh(threshold)

    # Optional mesh smoothing for better quality
    if config.mesh_params.get("smooth_iterations", 0) > 0:
        smooth_iters = config.mesh_params["smooth_iterations"]
        relaxation = config.mesh_params.get("smooth_relaxation", 0.1)
        threshold = threshold.smooth(n_iter=smooth_iters, relaxation_factor=relaxation)
        logger.info(f"Applied mesh smoothing with {smooth_iters} iterations")

    # Compute and log mesh statistics
    n_cells = threshold.n_cells
    n_points = threshold.n_points
    logger.info(f"Generated mesh with {n_cells} elements and {n_points} nodes")
    logger.info(f"Material distribution: {material_info}")

    return threshold, material_info


def _create_material_mapping(
    bone: np.ndarray, pore: np.ndarray
) -> Tuple[np.ndarray, Dict]:
    """Create material ID mapping for mesh

    Args:
        bone: Binary bone mask
        pore: Binary pore mask

    Returns:
        Tuple of (material_array, material_info)
    """
    materials = np.zeros_like(bone, dtype=np.uint8)

    # Material IDs:
    # 0: void (inactive)
    # 1: bone
    # 2: pore
    materials[bone] = 1
    materials[pore] = 2

    # Create material statistics
    total_elements = bone.size
    bone_elements = np.sum(bone)
    pore_elements = np.sum(pore)
    void_elements = total_elements - bone_elements - pore_elements

    material_info = {
        "bone_fraction": float(bone_elements / total_elements),
        "pore_fraction": float(pore_elements / total_elements),
        "void_fraction": float(void_elements / total_elements),
        "n_bone_elements": int(bone_elements),
        "n_pore_elements": int(pore_elements),
        "material_ids": {"bone": 1, "pore": 2, "void": 0},
    }

    return materials.flatten(order="F"), material_info
