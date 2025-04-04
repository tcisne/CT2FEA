# ct2fea/materials/linear.py
import numpy as np
from ...config import Config


class LinearMaterialMapper:
    def __init__(self, config: Config):
        self.min_density = config.material_params.get("min_density", 1.0)
        self.max_density = config.material_params.get("max_density", 2.0)
        self.E_coeff = config.material_params.get("E_coeff", (10000, 2.0))

    def calculate_properties(self, ct_values: np.ndarray) -> dict:
        densities = self.min_density + (self.max_density - self.min_density) * ct_values
        E = self.E_coeff[0] * (densities ** self.E_coeff[1])
        return {"density": densities, "youngs_modulus": E, "poissons_ratio": 0.3}
