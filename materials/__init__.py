# ct2fea/materials/__init__.py
from .linear import LinearMaterialMapper
from .nonlinear.plasticity import PlasticityMaterialMapper
from .gpu.material_mapping import GPUMaterialMapper


def get_material_mapper(config):
    """Enhanced GPU detection with fallback"""
    if config.use_gpu:
        try:
            # Try CUDA first
            from numba import cuda

            if cuda.is_available():
                from .gpu.material_mapping import GPUMaterialMapper

                return GPUMaterialMapper(config)
        except ImportError:
            pass

        # Fallback to OpenCL
        try:
            from ..acceleration import OpenCLManager
            from .gpu.opencl_material_mapper import (
                OpenCLMaterialMapper,
            )  # Needs implementation

            OpenCLManager().initialize()
            return OpenCLMaterialMapper(config)
        except Exception as e:
            logging.warning(f"GPU acceleration failed: {str(e)}")

    # Fallback to CPU models
    if config.material_model == "plasticity":
        return PlasticityMaterialMapper(config)
    if config.material_model == "hyperelastic":
        return HyperelasticMaterialMapper(config)
    return LinearMaterialMapper(config)
