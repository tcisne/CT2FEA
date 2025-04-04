import pytest
import numpy as np
import os
from pathlib import Path
from ..config import Config


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU acceleration")


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data"""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Create temporary directory for test outputs"""
    return tmp_path_factory.mktemp("test_output")


@pytest.fixture(scope="session")
def sample_ct_stack(test_data_dir):
    """Generate a sample CT stack for testing

    Creates a simple geometric phantom with bone-like structures
    """
    size = (50, 50, 50)
    phantom = np.zeros(size)

    # Add cylindrical structure
    center = np.array(size) // 2
    radius = 15
    for i in range(size[0]):
        for j in range(size[1]):
            dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if dist < radius:
                phantom[i, j, :] = 0.8

    # Add some internal variation
    noise = np.random.normal(0, 0.1, size)
    phantom += noise
    phantom = np.clip(phantom, 0, 1)

    # Save as TIFF stack
    import tifffile

    for i in range(size[2]):
        fname = test_data_dir / f"slice_{i:03d}.tiff"
        tifffile.imwrite(fname, (phantom[:, :, i] * 65535).astype(np.uint16))

    return phantom


@pytest.fixture
def config(test_data_dir, test_output_dir):
    """Create base configuration for testing"""
    return Config(
        input_folder=str(test_data_dir),
        output_dir=str(test_output_dir),
        voxel_size_um=10.0,
        material_model="linear",
        material_params={
            "min_density": 1.0,
            "max_density": 2.0,
            "E_coeff": (10000, 2.0),
            "yield_stress": 100.0,
            "hardening_coeff": 0.1,
        },
        denoise_params={"method": "gaussian", "sigma": 1.0},
        mesh_params={"smooth_iterations": 0, "smooth_relaxation": 0.1},
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if GPU is not available"""
    if not _check_gpu_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


def _check_gpu_available():
    """Check if GPU acceleration is available"""
    try:
        # Try CUDA first
        from numba import cuda

        return cuda.is_available()
    except ImportError:
        pass

    try:
        # Try OpenCL
        import pyopencl as cl

        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices(cl.device_type.GPU)
            if devices:
                return True
        return False
    except:
        return False
