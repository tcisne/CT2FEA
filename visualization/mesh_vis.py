# ct2fea/visualization/mesh_vis.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import pyvista as pv


def visualize_mesh(
    mesh: pv.UnstructuredGrid,
    output_dir: Path,
) -> str:
    """Visualize 3D mesh

    Args:
        mesh: PyVista mesh
        output_dir: Output directory

    Returns:
        Path to saved image
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plotter
    plotter = pv.Plotter(off_screen=True)

    # Add mesh to plotter
    plotter.add_mesh(
        mesh,
        show_edges=True,
        opacity=0.7,
        color="lightblue",
    )

    # Set camera position for a good view
    plotter.view_isometric()

    # Save screenshot
    output_path = output_dir / "mesh_visualization.png"
    plotter.screenshot(str(output_path), window_size=(1200, 1200))

    # Close plotter
    plotter.close()

    # Save additional views
    save_mesh_views(mesh, output_dir)

    return str(output_path)


def save_mesh_views(
    mesh: pv.UnstructuredGrid,
    output_dir: Path,
) -> None:
    """Save multiple views of the mesh

    Args:
        mesh: PyVista mesh
        output_dir: Output directory
    """
    views = {
        "front": (0, 0, 1),
        "top": (0, 1, 0),
        "side": (1, 0, 0),
    }

    for view_name, direction in views.items():
        # Create plotter
        plotter = pv.Plotter(off_screen=True)

        # Add mesh to plotter
        plotter.add_mesh(
            mesh,
            show_edges=True,
            opacity=0.7,
            color="lightblue",
        )

        # Set camera position
        plotter.view_vector(direction)

        # Save screenshot
        output_path = output_dir / f"mesh_{view_name}_view.png"
        plotter.screenshot(str(output_path), window_size=(1200, 1200))

        # Close plotter
        plotter.close()


def visualize_materials(
    mesh: pv.UnstructuredGrid,
    materials: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Visualize material properties on mesh

    Args:
        mesh: PyVista mesh
        materials: Material properties
        output_dir: Output directory

    Returns:
        Path to saved image
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a copy of the mesh with material properties
    mesh_with_props = mesh.copy()

    # Add material properties to mesh
    if "youngs_modulus" in materials:
        mesh_with_props.cell_data["Youngs_Modulus"] = materials["youngs_modulus"]
        property_name = "Youngs_Modulus"
        title = "Young's Modulus (MPa)"
    elif "density" in materials:
        mesh_with_props.cell_data["Density"] = materials["density"]
        property_name = "Density"
        title = "Density (g/cm³)"
    else:
        # Use first available property
        prop_name = list(materials.keys())[0]
        mesh_with_props.cell_data[prop_name] = materials[prop_name]
        property_name = prop_name
        title = prop_name.replace("_", " ").title()

    # Create plotter
    plotter = pv.Plotter(off_screen=True)

    # Add mesh to plotter with scalar bar
    plotter.add_mesh(
        mesh_with_props,
        scalars=property_name,
        show_edges=False,
        cmap="viridis",
        scalar_bar_args={"title": title},
    )

    # Set camera position for a good view
    plotter.view_isometric()

    # Save screenshot
    output_path = output_dir / "material_visualization.png"
    plotter.screenshot(str(output_path), window_size=(1200, 1200))

    # Close plotter
    plotter.close()

    # Create histogram of material properties
    create_material_histogram(materials, output_dir)

    return str(output_path)


def create_material_histogram(
    materials: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Create histogram of material properties

    Args:
        materials: Material properties
        output_dir: Output directory

    Returns:
        Path to saved image
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram of Young's modulus if available
    if "youngs_modulus" in materials:
        ax.hist(materials["youngs_modulus"].flatten(), bins=50, alpha=0.7)
        ax.set_xlabel("Young's Modulus (MPa)")
        title = "Young's Modulus Distribution"
        filename = "youngs_modulus_histogram.png"
    # Otherwise plot density if available
    elif "density" in materials:
        ax.hist(materials["density"].flatten(), bins=50, alpha=0.7)
        ax.set_xlabel("Density (g/cm³)")
        title = "Density Distribution"
        filename = "density_histogram.png"
    # Otherwise use first available property
    else:
        prop_name = list(materials.keys())[0]
        ax.hist(materials[prop_name].flatten(), bins=50, alpha=0.7)
        ax.set_xlabel(prop_name.replace("_", " ").title())
        title = f"{prop_name.replace('_', ' ').title()} Distribution"
        filename = f"{prop_name}_histogram.png"

    # Add title and labels
    ax.set_title(title)
    ax.set_ylabel("Frequency")

    # Save figure
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)
