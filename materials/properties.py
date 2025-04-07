# ct2fea/materials/properties.py
import numpy as np
from typing import Dict, Any
from ..config import Config


class LinearMaterialMapper:
    """Linear elastic material property mapper"""

    def __init__(self, config: Config):
        """Initialize with configuration

        Args:
            config: Configuration object
        """
        self.min_density = config.material_params.get("min_density", 1.0)
        self.max_density = config.material_params.get("max_density", 2.0)
        self.E_coeff = config.material_params.get("E_coeff", (10000, 2.0))
        self.poissons_ratio = config.material_params.get("poissons_ratio", 0.3)

    def calculate_properties(self, ct_values: np.ndarray) -> Dict[str, Any]:
        """Calculate material properties from CT values

        Args:
            ct_values: Normalized CT values (0-1)

        Returns:
            Dictionary with material properties
        """
        # Map CT values to density
        densities = self.min_density + (self.max_density - self.min_density) * ct_values

        # Calculate Young's modulus using power law relationship
        # E = a * (density^b)
        E = self.E_coeff[0] * (densities ** self.E_coeff[1])

        return {
            "density": densities,
            "youngs_modulus": E,
            "poissons_ratio": np.ones_like(E) * self.poissons_ratio,
        }


def get_material_mapper(config: Config) -> LinearMaterialMapper:
    """Get material mapper based on configuration

    Args:
        config: Configuration object

    Returns:
        Material mapper object
    """
    # In the simplified version, we only support linear elastic materials
    return LinearMaterialMapper(config)
