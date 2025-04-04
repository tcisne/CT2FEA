import pytest
import numpy as np
import pyvista as pv
from pathlib import Path
from ..meshing.voxel_mesher import create_voxel_mesh
from ..meshing.mesh_quality import check_mesh_quality
from ..config import Config


@pytest.fixture
def sample_masks():
    """Create sample bone and pore masks for testing"""
    size = 20
    bone = np.zeros((size, size, size), dtype=bool)
    pore = np.zeros((size, size, size), dtype=bool)

    # Create simple geometry
    bone[5:15, 5:15, 5:15] = True
    pore[12:17, 12:17, 12:17] = True
    bone[pore] = False  # Ensure no overlap

    return bone, pore


@pytest.fixture
def config():
    """Create test configuration"""
    return Config(
        input_folder="test_data",
        output_dir="test_output",
        voxel_size_um=10.0,
        material_model="linear",
    )


def test_create_voxel_mesh(sample_masks, config):
    """Test voxel mesh creation"""
    bone_mask, pore_mask = sample_masks
    mesh, material_info = create_voxel_mesh(bone_mask, pore_mask, config)

    # Check mesh type
    assert isinstance(mesh, pv.UnstructuredGrid)

    # Check material info
    assert "bone_fraction" in material_info
    assert "pore_fraction" in material_info
    assert 0 <= material_info["bone_fraction"] <= 1
    assert 0 <= material_info["pore_fraction"] <= 1

    # Check material array
    assert "Material" in mesh.cell_data
    assert np.all(mesh.cell_data["Material"] >= 0)


def test_mesh_quality(sample_masks, config):
    """Test mesh quality checking"""
    bone_mask, pore_mask = sample_masks
    mesh, _ = create_voxel_mesh(bone_mask, pore_mask, config)

    quality_metrics = check_mesh_quality(mesh)

    # Check required metrics
    assert "min_aspect_ratio" in quality_metrics
    assert "max_aspect_ratio" in quality_metrics
    assert "mean_aspect_ratio" in quality_metrics
    assert "jacobian_ok" in quality_metrics

    # Validate metric values
    assert 0 <= quality_metrics["min_aspect_ratio"] <= 1
    assert quality_metrics["min_aspect_ratio"] <= quality_metrics["max_aspect_ratio"]
    assert quality_metrics["jacobian_ok"] in (True, False)


def test_empty_mesh(config):
    """Test handling of empty masks"""
    empty_mask = np.zeros((10, 10, 10), dtype=bool)

    with pytest.raises(ValueError):
        create_voxel_mesh(empty_mask, empty_mask, config)
