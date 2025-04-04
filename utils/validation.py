# ct2fea/utils/validation.py
import numpy as np
import pyvista as pv
from typing import Dict, Any, Tuple, List
from pathlib import Path
import logging
import json
import datetime
from ..utils.errors import (
    CTDataError,
    SegmentationError,
    MeshingError,
    MaterialError,
    ConfigError,
)


def validate_ct_data(data: np.ndarray, metadata: Dict[str, Any]) -> None:
    """Validate CT data and metadata

    Args:
        data: CT volume data
        metadata: Associated metadata

    Raises:
        CTDataError: If validation fails
    """
    logger = logging.getLogger(__name__)

    # Check data type and dimensions
    if not isinstance(data, np.ndarray):
        raise CTDataError("CT data must be a numpy array")

    if data.ndim != 3:
        raise CTDataError("CT data must be 3-dimensional", f"Got shape: {data.shape}")

    # Check for NaN or infinite values
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise CTDataError("CT data contains NaN or infinite values")

    # Validate value range
    if data.min() < 0:
        logger.warning("CT data contains negative values")

    # Log data statistics
    logger.info(
        f"CT data statistics: shape={data.shape}, "
        f"range=[{data.min():.2f}, {data.max():.2f}], "
        f"mean={data.mean():.2f}, std={data.std():.2f}"
    )


def validate_segmentation(
    bone_mask: np.ndarray, pore_mask: np.ndarray, min_volume_fraction: float = 0.001
) -> None:
    """Validate segmentation results

    Args:
        bone_mask: Binary bone mask
        pore_mask: Binary pore mask
        min_volume_fraction: Minimum acceptable volume fraction

    Raises:
        SegmentationError: If validation fails
    """
    # Check mask types
    if bone_mask.dtype != np.bool_ or pore_mask.dtype != np.bool_:
        raise SegmentationError("Segmentation masks must be boolean arrays")

    # Check for overlapping regions
    if np.any(np.logical_and(bone_mask, pore_mask)):
        raise SegmentationError("Bone and pore masks overlap")

    # Check minimum volume fractions
    bone_fraction = np.mean(bone_mask)
    pore_fraction = np.mean(pore_mask)

    if bone_fraction < min_volume_fraction:
        raise SegmentationError(
            "Insufficient bone volume detected", f"Bone fraction: {bone_fraction:.4f}"
        )

    logging.getLogger(__name__).info(
        f"Segmentation fractions - Bone: {bone_fraction:.4f}, Pore: {pore_fraction:.4f}"
    )


def validate_mesh(mesh: pv.UnstructuredGrid, min_quality: float = 0.1) -> None:
    """Validate mesh quality

    Args:
        mesh: PyVista mesh
        min_quality: Minimum acceptable element quality

    Raises:
        MeshingError: If validation fails
    """
    logger = logging.getLogger(__name__)

    if mesh.n_cells == 0:
        raise MeshingError("Empty mesh generated")

    if mesh.n_points < 8:
        raise MeshingError("Insufficient mesh nodes")

    # Compute quality metrics
    quality = mesh.compute_cell_quality()
    min_qual = quality["CellQuality"].min()

    if min_qual < min_quality:
        raise MeshingError(
            f"Mesh contains poor quality elements (min quality: {min_qual:.3f})",
            f"Number of poor elements: {np.sum(quality['CellQuality'] < min_quality)}",
        )

    # Check for inverted elements
    if min_qual <= 0:
        raise MeshingError("Mesh contains inverted elements")

    logger.info(
        f"Mesh validation passed - Elements: {mesh.n_cells}, "
        f"Nodes: {mesh.n_points}, Min quality: {min_qual:.3f}"
    )


def validate_material_properties(
    properties: Dict[str, np.ndarray], material_model: str
) -> None:
    """Validate material property calculations

    Args:
        properties: Dictionary of material properties
        material_model: Material model type

    Raises:
        MaterialError: If validation fails
    """
    required_props = {
        "linear": ["density", "youngs_modulus", "poissons_ratio"],
        "plasticity": [
            "density",
            "youngs_modulus",
            "poissons_ratio",
            "yield_stress",
            "hardening_coeff",
        ],
        "hyperelastic": ["density", "c10", "d1"],
    }

    # Check required properties
    if material_model not in required_props:
        raise MaterialError(f"Unknown material model: {material_model}")

    missing = [p for p in required_props[material_model] if p not in properties]
    if missing:
        raise MaterialError(
            f"Missing required properties for {material_model} model",
            f"Missing: {', '.join(missing)}",
        )

    # Validate property ranges
    for name, prop in properties.items():
        if np.any(np.isnan(prop)) or np.any(np.isinf(prop)):
            raise MaterialError(f"Property '{name}' contains invalid values")
        if name != "poissons_ratio" and np.any(prop <= 0):
            raise MaterialError(f"Property '{name}' contains non-positive values")
        if name == "poissons_ratio" and np.any(np.abs(prop) >= 0.5):
            raise MaterialError("Invalid Poisson's ratio values detected")


def validate_output_path(path: Path) -> None:
    """Validate output path

    Args:
        path: Output directory path

    Raises:
        ConfigError: If validation fails
    """
    try:
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # Check write permissions
        test_file = path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise ConfigError(
                "Output directory is not writable", f"Path: {path}, Error: {str(e)}"
            )

    except Exception as e:
        raise ConfigError(
            "Failed to validate output directory", f"Path: {path}, Error: {str(e)}"
        )


class ValidationMetrics:
    """Calculate and store validation metrics for pipeline outputs"""

    def __init__(self, output_dir: Path):
        """Initialize validation metrics

        Args:
            output_dir: Output directory for saving metrics
        """
        self.output_dir = output_dir
        self.metrics: Dict[str, Dict[str, Any]] = {
            "ct_processing": {},
            "segmentation": {},
            "mesh": {},
            "materials": {},
            "overall": {},
        }
        self.logger = logging.getLogger(__name__)

    def add_ct_metrics(self, ct_data: np.ndarray) -> None:
        """Add CT processing metrics

        Args:
            ct_data: Processed CT data
        """
        metrics = {
            "shape": list(ct_data.shape),
            "min_value": float(np.min(ct_data)),
            "max_value": float(np.max(ct_data)),
            "mean_value": float(np.mean(ct_data)),
            "std_value": float(np.std(ct_data)),
            "histogram_bins": 10,
            "histogram": [float(x) for x in np.histogram(ct_data, bins=10)[0]],
        }

        self.metrics["ct_processing"] = metrics
        self.logger.info(f"Added CT processing metrics: {metrics}")

    def add_segmentation_metrics(self, segmentation_metrics: Dict[str, Any]) -> None:
        """Add segmentation metrics

        Args:
            segmentation_metrics: Metrics from segmentation validation
        """
        self.metrics["segmentation"] = segmentation_metrics
        self.logger.info(f"Added segmentation metrics: {segmentation_metrics}")

    def add_mesh_metrics(self, mesh: pv.UnstructuredGrid) -> None:
        """Add mesh quality metrics

        Args:
            mesh: PyVista mesh
        """
        quality = mesh.compute_cell_quality()

        metrics = {
            "n_cells": mesh.n_cells,
            "n_points": mesh.n_points,
            "min_quality": float(quality["CellQuality"].min()),
            "max_quality": float(quality["CellQuality"].max()),
            "mean_quality": float(quality["CellQuality"].mean()),
            "std_quality": float(quality["CellQuality"].std()),
            "quality_histogram": [
                float(x) for x in np.histogram(quality["CellQuality"], bins=10)[0]
            ],
            "n_poor_quality": int(np.sum(quality["CellQuality"] < 0.1)),
        }

        self.metrics["mesh"] = metrics
        self.logger.info(f"Added mesh metrics: {metrics}")

    def add_material_metrics(self, materials: Dict[str, np.ndarray]) -> None:
        """Add material property metrics

        Args:
            materials: Dictionary of material properties
        """
        metrics = {}

        for name, data in materials.items():
            metrics[name] = {
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
            }

        self.metrics["materials"] = metrics
        self.logger.info(f"Added material metrics: {metrics}")

    def calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall pipeline metrics

        Returns:
            Dictionary of overall metrics
        """
        # Calculate overall quality score (example implementation)
        overall_score = 0.0
        components = 0

        # CT quality (based on contrast)
        if (
            "ct_processing" in self.metrics
            and "std_value" in self.metrics["ct_processing"]
        ):
            ct_contrast = self.metrics["ct_processing"]["std_value"]
            ct_score = min(1.0, ct_contrast * 5)  # Normalize to 0-1
            overall_score += ct_score
            components += 1

        # Segmentation quality
        if (
            "segmentation" in self.metrics
            and "largest_bone_fraction" in self.metrics["segmentation"]
        ):
            seg_score = self.metrics["segmentation"]["largest_bone_fraction"]
            overall_score += seg_score
            components += 1

        # Mesh quality
        if "mesh" in self.metrics and "mean_quality" in self.metrics["mesh"]:
            mesh_score = self.metrics["mesh"]["mean_quality"]
            overall_score += mesh_score
            components += 1

        # Calculate final score
        if components > 0:
            overall_score /= components

        overall = {
            "quality_score": float(overall_score),
            "components_evaluated": components,
            "timestamp": str(datetime.datetime.now()),
        }

        self.metrics["overall"] = overall
        return overall

    def save_metrics(self) -> Path:
        """Save metrics to JSON file

        Returns:
            Path to saved metrics file
        """
        # Calculate overall metrics before saving
        self.calculate_overall_metrics()

        metrics_path = self.output_dir / "validation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        self.logger.info(f"Saved validation metrics to {metrics_path}")
        return metrics_path

    def generate_report(self) -> Path:
        """Generate HTML validation report

        Returns:
            Path to report file
        """
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        # Create quality visualizations
        plt.figure(figsize=(10, 6))

        # Mesh quality histogram
        if "mesh" in self.metrics and "quality_histogram" in self.metrics["mesh"]:
            plt.bar(range(10), self.metrics["mesh"]["quality_histogram"])
            plt.title("Mesh Quality Distribution")
            plt.xlabel("Quality Bin")
            plt.ylabel("Count")

            # Save plot to memory
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("ascii")
            mesh_plot = (
                f'<img src="data:image/png;base64,{img_str}" alt="Mesh Quality">'
            )
        else:
            mesh_plot = "<p>No mesh quality data available</p>"

        plt.close()

        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CT2FEA Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 10px 0; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                .section {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>CT2FEA Validation Report</h1>
            
            <div class="section">
                <h2>Overall Quality</h2>
                <div class="metric">Quality Score: <span class="{self._get_quality_class(self.metrics["overall"].get("quality_score", 0))}">{self.metrics["overall"].get("quality_score", 0):.2f}</span></div>
            </div>
            
            <div class="section">
                <h2>Mesh Quality</h2>
                {mesh_plot}
                <div class="metric">Cells: {self.metrics["mesh"].get("n_cells", "N/A")}</div>
                <div class="metric">Points: {self.metrics["mesh"].get("n_points", "N/A")}</div>
                <div class="metric">Mean Quality: {self.metrics["mesh"].get("mean_quality", "N/A"):.3f}</div>
                <div class="metric">Poor Quality Elements: <span class="{self._get_poor_elements_class(self.metrics["mesh"].get("n_poor_quality", 0), self.metrics["mesh"].get("n_cells", 1))}">{self.metrics["mesh"].get("n_poor_quality", "N/A")}</span></div>
            </div>
            
            <div class="section">
                <h2>Segmentation</h2>
                <div class="metric">Bone Fraction: {self.metrics["segmentation"].get("bone_fraction", "N/A"):.3f}</div>
                <div class="metric">Pore Fraction: {self.metrics["segmentation"].get("pore_fraction", "N/A"):.3f}</div>
                <div class="metric">Bone Components: {self.metrics["segmentation"].get("bone_components", "N/A")}</div>
            </div>
            
            <div class="section">
                <h2>Material Properties</h2>
                <table border="1" cellpadding="5">
                    <tr>
                        <th>Property</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean</th>
                        <th>Std</th>
                    </tr>
                    {self._generate_material_table()}
                </table>
            </div>
        </body>
        </html>
        """

        report_path = self.output_dir / "validation_report.html"
        with open(report_path, "w") as f:
            f.write(html)

        self.logger.info(f"Generated validation report at {report_path}")
        return report_path

    def _get_quality_class(self, score: float) -> str:
        """Get CSS class based on quality score"""
        if score > 0.8:
            return "good"
        elif score > 0.5:
            return "warning"
        else:
            return "error"

    def _get_poor_elements_class(self, poor_elements: int, total_elements: int) -> str:
        """Get CSS class based on poor elements percentage"""
        if total_elements == 0:
            return "error"

        percentage = poor_elements / total_elements
        if percentage < 0.01:  # Less than 1%
            return "good"
        elif percentage < 0.05:  # Less than 5%
            return "warning"
        else:
            return "error"

    def _generate_material_table(self) -> str:
        """Generate HTML table for material properties"""
        rows = []

        if "materials" in self.metrics:
            for name, props in self.metrics["materials"].items():
                row = f"""
                <tr>
                    <td>{name}</td>
                    <td>{props.get("min", "N/A"):.3f}</td>
                    <td>{props.get("max", "N/A"):.3f}</td>
                    <td>{props.get("mean", "N/A"):.3f}</td>
                    <td>{props.get("std", "N/A"):.3f}</td>
                </tr>
                """
                rows.append(row)

        if not rows:
            rows = ["<tr><td colspan='5'>No material data available</td></tr>"]

        return "\n".join(rows)
