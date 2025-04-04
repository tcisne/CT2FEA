# ct2fea/utils/interactive_vis.py
import pyvista as pv
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import logging
from ..config import Config

logger = logging.getLogger(__name__)


class InteractivePlotter:
    def __init__(self, config: Config):
        """Initialize interactive plotter with configuration"""
        self.config = config
        self.plotter = pv.Plotter()
        self._setup_plotter()

    def _setup_plotter(self) -> None:
        """Configure base plotter settings"""
        self.plotter.set_background("white")
        self.plotter.enable_anti_aliasing()
        self.plotter.add_axes()
        self.plotter.enable_depth_peeling()

    def visualize_mesh(
        self,
        mesh: pv.UnstructuredGrid,
        scalar_field: str = "Material",
        show_edges: bool = True,
        cmap: str = "viridis",
        save_path: Optional[Path] = None,
    ) -> None:
        """Visualize mesh with interactive controls

        Args:
            mesh: PyVista mesh object
            scalar_field: Name of scalar field to color by
            show_edges: Whether to show mesh edges
            cmap: Colormap name
            save_path: Optional path to save screenshot
        """
        # Add mesh to scene
        self.plotter.add_mesh(
            mesh,
            scalars=scalar_field,
            show_edges=show_edges,
            edge_color="black",
            line_width=0.5,
            cmap=cmap,
        )

        # Add interactive mesh clipping widgets
        self.plotter.add_plane_widget(
            callback=lambda normal, origin: self._clip_mesh(mesh, normal, origin),
            assign_to_axis="z",
        )

        # Add scalar bar
        self.plotter.add_scalar_bar(
            title=scalar_field,
            n_labels=5,
            italic=False,
            shadow=True,
            fmt="%.2f",
            font_family="arial",
        )

        if save_path:
            self.plotter.show(screenshot=str(save_path))
        else:
            self.plotter.show()

    def create_volume_viewer(
        self,
        volume: np.ndarray,
        spacing: Optional[List[float]] = None,
        cmap: str = "viridis",
    ) -> None:
        """Create interactive volume viewer

        Args:
            volume: 3D numpy array
            spacing: Optional voxel spacing
            cmap: Colormap name
        """
        if spacing is None:
            spacing = [self.config.voxel_size_um / 1000.0] * 3

        # Create PyVista uniform grid
        grid = pv.UniformGrid()
        grid.dimensions = np.array(volume.shape) + 1
        grid.spacing = spacing
        grid.cell_data["values"] = volume.flatten(order="F")

        # Add volume to scene with opacity transfer function
        self.plotter.add_volume(
            grid,
            cmap=cmap,
            opacity="sigmoid",
            shade=True,
            ambient=0.3,
            diffuse=0.7,
            specular=0.5,
        )

        # Add orthogonal slicers
        self.plotter.add_volume_clip_plane()

        self.plotter.show()

    def visualize_material_distribution(
        self,
        mesh: pv.UnstructuredGrid,
        materials: Dict[str, np.ndarray],
        save_path: Optional[Path] = None,
    ) -> None:
        """Create interactive material property visualization

        Args:
            mesh: PyVista mesh object
            materials: Dictionary of material properties
            save_path: Optional path to save screenshot
        """
        # Create multi-view plotter
        plotter = pv.Plotter(shape=(2, 2))

        # Plot different material properties
        for i, (name, data) in enumerate(materials.items()):
            row, col = divmod(i, 2)
            plotter.subplot(row, col)
            plotter.add_mesh(
                mesh,
                scalars=data.ravel(),
                show_edges=True,
                edge_color="black",
                line_width=0.5,
                scalar_bar_args={"title": name.replace("_", " ").title()},
            )
            plotter.add_axes()

        if save_path:
            plotter.show(screenshot=str(save_path))
        else:
            plotter.show()

    def _clip_mesh(
        self, mesh: pv.UnstructuredGrid, normal: List[float], origin: List[float]
    ) -> None:
        """Callback for interactive mesh clipping

        Args:
            mesh: PyVista mesh object
            normal: Clipping plane normal vector
            origin: Clipping plane origin point
        """
        self.plotter.clear()
        clipped = mesh.clip(normal=normal, origin=origin)
        self.plotter.add_mesh(clipped)
        self.plotter.add_axes()
