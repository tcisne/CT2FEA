import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..config import Config


def visualize_slice(
    data: np.ndarray,
    output_dir: Path,
    config: Config,
    name: str = "",
    overlay: Optional[np.ndarray] = None,
    colormap: str = "gray",
) -> None:
    """Visualize orthogonal slices of 3D volume with optional overlay

    Args:
        data: 3D volume data
        output_dir: Output directory
        config: Configuration object
        name: Name prefix for output file
        overlay: Optional binary mask overlay
        colormap: Matplotlib colormap name
    """
    logger = logging.getLogger(__name__)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get central slices
    slices = [data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2]
    views = ["Axial", "Coronal", "Sagittal"]
    slice_funcs = [
        lambda d, i: d[i],
        lambda d, i: d[:, i, :].T,
        lambda d, i: d[:, :, i].T,
    ]

    for ax, slc, title, slice_func in zip(axes, slices, views, slice_funcs):
        im = ax.imshow(slice_func(data, slc), cmap=colormap)
        if overlay is not None:
            overlay_slice = slice_func(overlay, slc)
            ax.imshow(overlay_slice, cmap="Reds", alpha=0.3)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        ax.set_title(f"{title} View")
        ax.axis("off")

    plt.tight_layout()
    prefix = f"{name}_" if name else ""
    output_path = output_dir / f"{prefix}ct_slices.png"
    plt.savefig(output_path, dpi=config.visualization_dpi)
    plt.close()
    logger.info(f"Saved CT slices visualization to {output_path}")


def visualize_mesh(
    mesh: pv.UnstructuredGrid, output_dir: Path, config: Config, show_edges: bool = True
) -> None:
    """Visualize 3D mesh with interactive controls and multiple viewpoints

    Args:
        mesh: PyVista mesh
        output_dir: Output directory
        config: Configuration settings
        show_edges: Whether to show mesh edges
    """
    if not config.save_visualization:
        return

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        mesh,
        scalars="Material",
        cmap="viridis",
        show_edges=show_edges,
        edge_color="black",
        line_width=0.5,
    )
    plotter.add_axes()
    plotter.camera_position = "iso"

    # Save isometric view
    output_path = output_dir / "mesh_iso.png"
    plotter.show(screenshot=output_path)

    # Save orthogonal views
    views = ["xy", "xz", "yz"]
    for view in views:
        plotter.camera_position = view
        output_path = output_dir / f"mesh_{view}.png"
        plotter.show(screenshot=output_path)

    plotter.close()


def plot_material_distribution(
    materials: Dict[str, np.ndarray], output_dir: Path, config: Config
) -> None:
    """Plot material property distributions

    Args:
        materials: Dictionary of material properties
        output_dir: Output directory
        config: Configuration settings
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (name, data) in enumerate(materials.items()):
        if i >= len(axes):
            break

        sns.histplot(data.ravel(), bins=50, ax=axes[i])
        axes[i].set_title(f"{name.replace('_', ' ').title()} Distribution")
        axes[i].set_xlabel(name)
        axes[i].set_ylabel("Count")

    plt.tight_layout()
    output_path = output_dir / "material_distributions.png"
    plt.savefig(output_path, dpi=config.visualization_dpi)
    plt.close()


def create_quality_plots(
    quality_metrics: Dict[str, float],
    mesh_data: Dict[str, np.ndarray],
    output_dir: Path,
    config: Config,
) -> None:
    """Create comprehensive mesh quality visualization

    Args:
        quality_metrics: Dictionary of quality metrics
        mesh_data: Dictionary containing mesh statistics
        output_dir: Output directory
        config: Configuration settings
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 3)

    # Element quality distribution
    ax1 = fig.add_subplot(gs[0, :2])
    sns.histplot(mesh_data["quality"], bins=50, ax=ax1)
    ax1.set_title("Element Quality Distribution")
    ax1.set_xlabel("Quality Metric")
    ax1.set_ylabel("Count")

    # Material distribution pie chart
    ax2 = fig.add_subplot(gs[0, 2])
    materials = ["Bone", "Pore"]
    sizes = [mesh_data["bone_fraction"], mesh_data["pore_fraction"]]
    ax2.pie(sizes, labels=materials, autopct="%1.1f%%")
    ax2.set_title("Material Distribution")

    # Quality metrics table
    ax3 = fig.add_subplot(gs[1, :])
    cell_text = [[f"{v:.3f}" for v in quality_metrics.values()]]
    ax3.table(
        cellText=cell_text,
        colLabels=quality_metrics.keys(),
        loc="center",
        cellLoc="center",
    )
    ax3.axis("off")
    ax3.set_title("Mesh Quality Metrics")

    plt.tight_layout()
    output_path = output_dir / "mesh_quality_analysis.png"
    plt.savefig(output_path, dpi=config.visualization_dpi)
    plt.close()
