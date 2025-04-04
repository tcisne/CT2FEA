# ct2fea/materials/nonlinear/plasticity.py
import numpy as np
from ...config import Config


class PlasticityMaterialMapper:
    def __init__(self, config: Config):
        self.config = config
        self.linear = LinearMaterialMapper(config)

    def calculate_properties(self, ct_values: np.ndarray) -> dict:
        linear_props = self.linear.calculate_properties(ct_values)
        return {
            **linear_props,
            "yield_stress": self._calc_yield_stress(linear_props["density"]),
            "hardening_coeff": self.config.nonlinear_params["hardening_coeff"],
        }

    def _calc_yield_stress(self, density: np.ndarray) -> np.ndarray:
        return self.config.nonlinear_params["yield_stress"] * np.sqrt(density)
