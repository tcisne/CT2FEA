import pytest
import numpy as np
from ..materials.linear import LinearMaterialMapper
from ..materials.nonlinear.plasticity import PlasticityMaterialMapper
from ..materials.gpu.material_mapping import GPUMaterialMapper
from ..config import Config


@pytest.fixture
def sample_ct_data():
    """Create sample CT data for testing"""
    return np.linspace(0, 1, 1000).reshape(10, 10, 10)


@pytest.fixture
def config():
    """Create test configuration"""
    return Config(
        input_folder="test_data",
        output_dir="test_output",
        voxel_size_um=10.0,
        material_model="linear",
        material_params={
            "min_density": 1.0,
            "max_density": 2.0,
            "E_coeff": (10000, 2.0),
            "yield_stress": 100.0,
            "hardening_coeff": 0.1,
        },
    )


def test_linear_material(sample_ct_data, config):
    """Test linear material property calculation"""
    mapper = LinearMaterialMapper(config)
    properties = mapper.calculate_properties(sample_ct_data)

    assert "density" in properties
    assert "youngs_modulus" in properties
    assert "poissons_ratio" in properties

    # Check value ranges
    assert np.all(properties["density"] >= config.material_params["min_density"])
    assert np.all(properties["density"] <= config.material_params["max_density"])
    assert np.all(properties["youngs_modulus"] > 0)
    assert np.all(properties["poissons_ratio"] > 0)


def test_plasticity_material(sample_ct_data, config):
    """Test plasticity material property calculation"""
    config.material_model = "plasticity"
    mapper = PlasticityMaterialMapper(config)
    properties = mapper.calculate_properties(sample_ct_data)

    # Check additional plasticity properties
    assert "yield_stress" in properties
    assert "hardening_coeff" in properties
    assert np.all(properties["yield_stress"] > 0)
    assert properties["hardening_coeff"] == config.material_params["hardening_coeff"]


@pytest.mark.gpu
def test_gpu_material(sample_ct_data, config):
    """Test GPU-accelerated material calculation"""
    try:
        mapper = GPUMaterialMapper(config)
        properties = mapper.calculate_properties(sample_ct_data)

        # Compare with CPU calculation
        cpu_mapper = LinearMaterialMapper(config)
        cpu_properties = cpu_mapper.calculate_properties(sample_ct_data)

        # Check results match within tolerance
        np.testing.assert_allclose(
            properties["density"], cpu_properties["density"], rtol=1e-5
        )
        np.testing.assert_allclose(
            properties["youngs_modulus"], cpu_properties["youngs_modulus"], rtol=1e-5
        )
    except ImportError:
        pytest.skip("GPU acceleration not available")
