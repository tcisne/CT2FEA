import pytest
import numpy as np
from ..processing.ct_processing import normalize_ct_volume, coarsen_volume
from ..processing.segmentation import segment_volume, validate_segmentation
from ..config import Config


@pytest.fixture
def sample_volume():
    """Create a sample CT volume for testing"""
    return np.random.normal(100, 20, (50, 50, 50))


@pytest.fixture
def config():
    """Create a test configuration"""
    return Config(
        input_folder="test_data",
        output_dir="test_output",
        voxel_size_um=10.0,
        material_model="linear",
    )


def test_normalize_ct_volume(sample_volume):
    """Test CT volume normalization"""
    normalized, stats = normalize_ct_volume(sample_volume, (1, 99))
    assert normalized.min() >= 0
    assert normalized.max() <= 1
    assert "clip_min" in stats
    assert "clip_max" in stats


def test_coarsen_volume(sample_volume):
    """Test volume coarsening"""
    factor = 2
    coarsened = coarsen_volume(sample_volume, factor)
    expected_shape = tuple(s // factor for s in sample_volume.shape)
    assert coarsened.shape == expected_shape


def test_segmentation(sample_volume, config):
    """Test bone and pore segmentation"""
    # Normalize first
    normalized, _ = normalize_ct_volume(sample_volume, (1, 99))

    # Segment
    bone_mask, pore_mask = segment_volume(normalized, config)

    # Check masks are binary
    assert bone_mask.dtype == bool
    assert pore_mask.dtype == bool

    # Check no overlap
    assert not np.any(np.logical_and(bone_mask, pore_mask))


def test_validate_segmentation():
    """Test segmentation validation"""
    bone = np.zeros((10, 10, 10), dtype=bool)
    pore = np.zeros((10, 10, 10), dtype=bool)

    # Set some bone and pore regions
    bone[2:5, 2:5, 2:5] = True
    pore[6:9, 6:9, 6:9] = True

    # Should pass
    validate_segmentation(bone, pore)

    # Should raise error on overlap
    bone[7, 7, 7] = True
    with pytest.raises(ValueError):
        validate_segmentation(bone, pore)
