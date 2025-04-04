import numpy as np
from ...config import Config


class HyperelasticMaterialMapper:
    """Implement hyperelastic material model"""

    def __init__(self, config: Config):
        self.config = config
        self.base_mapper = LinearMaterialMapper(config)

    def calculate_properties(self, ct_values: np.ndarray) -> dict:
        base_props = self.base_mapper.calculate_properties(ct_values)
        return {
            **base_props,
            "c10": self._calc_c10(base_props["density"]),
            "d1": self._calc_d1(base_props["density"]),
        }

    def _calc_c10(self, density: np.ndarray) -> np.ndarray:
        return 0.5 * density**1.5

    def _calc_d1(self, density: np.ndarray) -> np.ndarray:
        return 0.01 / density
