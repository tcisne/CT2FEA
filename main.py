# ct2fea/main.py
from pathlib import Path
import logging
import time
from typing import Dict, Any, Optional
import pyvista as pv
from tqdm import tqdm

from .config import Config
from .io.file_io import load_ct_stack, save_processed_ct
from .processing import (
    normalize_ct_volume,
    coarsen_volume,
    denoise_volume,
    segment_volume,
    validate_segmentation,
)
from .meshing import create_voxel_mesh, export_to_abaqus
from .materials import get_material_mapper
from .utils.logging import setup_logging, PipelineLogger
from .utils import visualization
from .utils.validation import (
    validate_ct_data,
    validate_mesh,
    validate_material_properties,
    validate_output_path,
)
from .utils.profiling import (
    PerformanceMonitor,
    profile_stage,
    check_memory_available,
    estimate_memory_usage,
)
from .utils.errors import CT2FEAError


class Pipeline:
    """Main CT2FEA processing pipeline with performance monitoring"""

    def __init__(self, config: Config):
        """Initialize pipeline with configuration

        Args:
            config: Configuration object
        """
        self.config = config
        self.stats: Dict[str, Any] = {}
        self._setup()
        self.interactive_vis = None
        if config.visualization_dpi > 0:  # Only create if visualization is enabled
            from .utils.interactive_vis import InteractivePlotter

            self.interactive_vis = InteractivePlotter(config)

    def _setup(self) -> None:
        """Set up logging, profiling, and output directory"""
        self.output_dir = Path(self.config.output_dir)
        validate_output_path(self.output_dir)

        setup_logging(self.output_dir)
        self.logger = PipelineLogger(__name__)

        # Initialize performance monitoring
        self.monitor = PerformanceMonitor(self.output_dir)

    def run(self) -> None:
        """Run the complete pipeline with performance monitoring"""
        try:
            self.monitor.start_profiling()
            start_time = time.time()

            self.logger.info("Starting CT2FEA Pipeline")
            self.config.validate()

            # Process CT data
            ct_volume = self._process_ct()

            # Check memory for mesh generation
            mesh_memory = self._estimate_mesh_memory(ct_volume.shape)
            if not check_memory_available(mesh_memory):
                raise CT2FEAError(
                    "Insufficient memory for mesh generation",
                    f"Required: {mesh_memory:.1f}MB",
                )

            # Generate mesh
            mesh = self._generate_mesh()

            # Calculate material properties
            materials = self._assign_materials(mesh)

            # Export results
            self._export_results(mesh, materials)

            # Generate reports
            self._generate_reports()

            elapsed_time = time.time() - start_time
            self.logger.info(f"Pipeline completed in {elapsed_time:.1f} seconds")

            # Stop profiling and generate performance report
            self.monitor.stop_profiling()
            self.monitor.generate_report()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    @profile_stage()
    def _process_ct(self) -> np.ndarray:
        """Process CT volume data with performance monitoring"""
        self.logger.start_stage("CT Processing")

        # Load and validate CT stack
        ct_volume, file_list = load_ct_stack(Path(self.config.input_folder))
        self.stats["n_slices"] = len(file_list)
        validate_ct_data(ct_volume, {"n_slices": len(file_list)})

        # Show interactive volume visualization if enabled
        if self.interactive_vis:
            self.interactive_vis.create_volume_viewer(ct_volume, name="original")

        # Create processing visualizations
        visualization.visualize_slice(
            ct_volume, self.output_dir, self.config, name="original"
        )

        # Normalize with parallel processing
        ct_norm, clip_stats = normalize_ct_volume(
            ct_volume, self.config.clip_percentiles
        )
        self.stats["normalization"] = clip_stats

        # Optional coarsening
        if self.config.coarsen_factor > 1:
            ct_norm = coarsen_volume(ct_norm, self.config.coarsen_factor)
            self.logger.info(f"Coarsened volume by factor {self.config.coarsen_factor}")

        # Denoising if enabled with parallel processing
        if self.config.denoise_method:
            ct_norm = denoise_volume(ct_norm, self.config.denoise_method, self.config)
            if self.interactive_vis:
                self.interactive_vis.create_volume_viewer(ct_norm, name="denoised")

        self.stats["ct_processed"] = ct_norm
        self.logger.end_stage()
        return ct_norm

    @profile_stage()
    def _generate_mesh(self) -> pv.UnstructuredGrid:
        """Generate and validate mesh with performance monitoring"""
        self.logger.start_stage("Mesh Generation")

        # Segmentation with parallel processing
        bone_mask, pore_mask = segment_volume(self.stats["ct_processed"], self.config)
        validate_segmentation(bone_mask, pore_mask)

        # Create mesh
        mesh, material_info = create_voxel_mesh(bone_mask, pore_mask, self.config)
        self.stats["mesh_info"] = material_info

        # Validate mesh quality
        validate_mesh(mesh)

        # Interactive mesh visualization if enabled
        if self.interactive_vis:
            self.interactive_vis.visualize_mesh(mesh)

        self.logger.end_stage(extra_data=material_info)
        return mesh

    @profile_stage()
    def _assign_materials(self, mesh: pv.UnstructuredGrid) -> Dict:
        """Calculate material properties with performance monitoring"""
        self.logger.start_stage("Material Assignment")

        mapper = get_material_mapper(self.config)
        materials = mapper.calculate_properties(self.stats["ct_processed"])

        # Validate properties
        validate_material_properties(materials, self.config.material_model)

        # Interactive material visualization if enabled
        if self.interactive_vis:
            self.interactive_vis.visualize_material_distribution(mesh, materials)

        self.logger.end_stage()
        return materials

    @profile_stage()
    def _export_results(self, mesh: pv.UnstructuredGrid, materials: Dict) -> None:
        """Export results with performance monitoring"""
        self.logger.start_stage("Export")

        # Save processed CT data
        save_processed_ct(self.output_dir, self.stats["ct_processed"], self.config)

        # Export Abaqus input file
        output_path = self.output_dir / "model.inp"
        export_to_abaqus(mesh, materials, self.config, output_path)

        self.logger.end_stage()

    def _generate_reports(self) -> None:
        """Generate quality and performance reports"""
        self.logger.start_stage("Report Generation")

        visualization.create_quality_plots(
            self.stats["quality_metrics"],
            self.stats["mesh_info"],
            self.output_dir,
            self.config,
        )

        self.logger.end_stage()

    def _estimate_mesh_memory(self, volume_shape: tuple) -> float:
        """Estimate memory requirements for mesh generation

        Args:
            volume_shape: Shape of CT volume

        Returns:
            Estimated memory requirement in MB
        """
        # Estimate number of elements (worst case)
        max_elements = np.prod(volume_shape)

        # Memory for node coordinates (float64)
        node_memory = estimate_memory_usage((max_elements * 8, 3), np.float64)

        # Memory for connectivity (int32)
        connect_memory = estimate_memory_usage((max_elements, 8), np.int32)

        # Memory for material properties (float32)
        material_memory = estimate_memory_usage((max_elements, 3), np.float32)

        # Add 20% overhead for other data structures
        total_memory = (node_memory + connect_memory + material_memory) * 1.2

        return total_memory


def main(config: Optional[Config] = None) -> None:
    """Main entry point with optional configuration

    Args:
        config: Optional configuration object. If None, will use GUI.
    """
    if config is None:
        from .io.gui import get_gui_inputs

        config = get_gui_inputs()
        if not config:
            return

    pipeline = Pipeline(config)
    pipeline.run()
