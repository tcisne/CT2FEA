import numpy as np
import pyvista as pv
import logging
from typing import Dict
from pathlib import Path
import jinja2
import matplotlib.pyplot as plt
import seaborn as sns


def check_mesh_quality(mesh: pv.UnstructuredGrid) -> Dict:
    """Calculate comprehensive mesh quality metrics

    Args:
        mesh: PyVista unstructured grid mesh
    Returns:
        Dictionary containing quality metrics
    """
    logger = logging.getLogger(__name__)
    qual = mesh.compute_cell_quality()

    # Calculate detailed metrics
    metrics = {
        "min_aspect_ratio": float(qual["CellQuality"].min()),
        "max_aspect_ratio": float(qual["CellQuality"].max()),
        "mean_aspect_ratio": float(qual["CellQuality"].mean()),
        "std_aspect_ratio": float(qual["CellQuality"].std()),
        "median_aspect_ratio": float(np.median(qual["CellQuality"])),
        "jacobian_ok": bool(np.all(qual["CellQuality"] > 0)),
        "n_poor_quality": int(np.sum(qual["CellQuality"] < 0.1)),
        "total_elements": len(qual["CellQuality"]),
    }

    if not metrics["jacobian_ok"]:
        logger.warning("Mesh contains inverted elements with negative Jacobian")
    if metrics["n_poor_quality"] > 0:
        logger.warning(f"Found {metrics['n_poor_quality']} elements with poor quality")

    return metrics


def generate_quality_report(metrics: Dict, output_dir: Path) -> None:
    """Generate HTML mesh quality report with visualizations

    Args:
        metrics: Dictionary of mesh quality metrics
        output_dir: Output directory for report
    """
    # Create quality distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(metrics["quality_distribution"], bins=50)
    plt.title("Element Quality Distribution")
    plt.xlabel("Quality Metric")
    plt.ylabel("Count")
    plot_path = output_dir / "quality_distribution.png"
    plt.savefig(plot_path)
    plt.close()

    # Generate HTML report
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mesh Quality Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .metric { margin: 10px 0; }
            .warning { color: red; }
        </style>
    </head>
    <body>
        <h1>Mesh Quality Report</h1>
        
        <h2>Summary Statistics</h2>
        <div class="metric">Total Elements: {{ metrics.total_elements }}</div>
        <div class="metric">Mean Aspect Ratio: {{ "%.3f"|format(metrics.mean_aspect_ratio) }}</div>
        <div class="metric">Median Aspect Ratio: {{ "%.3f"|format(metrics.median_aspect_ratio) }}</div>
        <div class="metric">Standard Deviation: {{ "%.3f"|format(metrics.std_aspect_ratio) }}</div>
        
        <h2>Quality Metrics</h2>
        <div class="metric">Minimum Quality: {{ "%.3f"|format(metrics.min_aspect_ratio) }}</div>
        <div class="metric">Maximum Quality: {{ "%.3f"|format(metrics.max_aspect_ratio) }}</div>
        <div class="metric">Poor Quality Elements: {{ metrics.n_poor_quality }}</div>
        
        {% if not metrics.jacobian_ok %}
        <div class="warning">Warning: Mesh contains inverted elements!</div>
        {% endif %}
        
        <h2>Quality Distribution</h2>
        <img src="quality_distribution.png" alt="Quality Distribution">
    </body>
    </html>
    """

    # Render and save HTML
    html = jinja2.Template(template).render(metrics=metrics)
    report_path = output_dir / "mesh_quality_report.html"
    report_path.write_text(html)

    logging.getLogger(__name__).info(f"Generated mesh quality report at {report_path}")
