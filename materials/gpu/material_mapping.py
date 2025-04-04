# ct2fea/materials/gpu/material_mapping.py
from numba import cuda
import numpy as np
from ...config import Config


class GPUMaterialMapper:
    def __init__(self, config: Config):
        self.block_size = config.gpu_params["block_size"]

    def calculate_properties(self, ct_values: np.ndarray) -> dict:
        ct_gpu = cuda.to_device(ct_values.ravel())
        densities_gpu = cuda.device_array_like(ct_gpu)
        E_gpu = cuda.device_array_like(ct_gpu)

        blocks = (ct_gpu.size + self.block_size - 1) // self.block_size
        self._gpu_kernel[blocks, self.block_size](ct_gpu, densities_gpu, E_gpu)

        return {
            "density": densities_gpu.copy_to_host().reshape(ct_values.shape),
            "youngs_modulus": E_gpu.copy_to_host().reshape(ct_values.shape),
            "poissons_ratio": 0.3,
        }

    @cuda.jit
    def _gpu_kernel(ct, density, E):
        i = cuda.grid(1)
        if i < ct.size:
            density[i] = 1.0 + ct[i] * 1.0
            E[i] = 10000.0 * (density[i] ** 2.0)
