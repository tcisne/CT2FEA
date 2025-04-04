# ct2fea/main.py
from pathlib import Path
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, Iterator
import pyvista as pv
from tqdm import tqdm

from .config import Config
from .io.file_io import (
    load_ct_stack,
    save_processed_ct,
    create_ct_iterator,
    get_ct_metadata,
    save_ct_chunks,
)
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
from .utils.errors import CT2FEAError, RecoverableError
from .utils.validation import ValidationMetrics


class Pipeline:
    """Main CT2FEA processing pipeline with performance monitoring and error recovery"""

    def __init__(self, config: Config, resume_from_checkpoint: bool = False):
        """Initialize pipeline with configuration

        Args:
            config: Configuration object
            resume_from_checkpoint: Whether to attempt resuming from checkpoint
        """
        self.config = config
        self.stats: Dict[str, Any] = {}
        self.resume_from_checkpoint = resume_from_checkpoint
        self._setup()

        # Initialize visualization if enabled
        self.interactive_vis = None
        if config.visualization_dpi > 0:  # Only create if visualization is enabled
            from .utils.interactive_vis import InteractivePlotter

            self.interactive_vis = InteractivePlotter(config)

    def _setup(self) -> None:
        """Set up logging, profiling, checkpoint management and output directory"""
        self.output_dir = Path(self.config.output_dir)
        validate_output_path(self.output_dir)

        setup_logging(self.output_dir)
        self.logger = PipelineLogger(__name__)

        # Initialize performance monitoring
        self.monitor = PerformanceMonitor(self.output_dir)

        # Initialize validation metrics
        self.validation = ValidationMetrics(self.output_dir)

        # Initialize checkpoint manager for error recovery
        from .utils.errors import CheckpointManager

        self.checkpoint_manager = CheckpointManager(self.output_dir)

        # Set up streaming parameters
        self.chunk_size = self.config.chunk_size or 10  # Default to 10 slices per chunk

    def run(self) -> None:
        """Run the complete pipeline with performance monitoring and error recovery"""
        try:
            self.monitor.start_profiling()
            start_time = time.time()

            self.logger.info("Starting CT2FEA Pipeline")
            self.config.validate()

            # Check for existing checkpoint if resuming
            checkpoint_data = None
            if self.resume_from_checkpoint:
                checkpoint_data = self.checkpoint_manager.load_checkpoint()
                if checkpoint_data:
                    self.logger.info(
                        f"Resuming from checkpoint at stage: {checkpoint_data['stage']}"
                    )
                else:
                    self.logger.info("No checkpoint found, starting from beginning")

            # Process CT data with streaming approach
            ct_metadata = self._get_ct_metadata()
            self.stats.update(ct_metadata)

            # Process CT data in chunks
            if not checkpoint_data or checkpoint_data["stage"] == "init":
                self._process_ct_streaming()
                self.checkpoint_manager.save_checkpoint(
                    "ct_processing", {"completed": True}
                )

            # Check memory for mesh generation
            volume_shape = (
                ct_metadata["n_slices"],
                ct_metadata["height"],
                ct_metadata["width"],
            )
            mesh_memory = self._estimate_mesh_memory(volume_shape)
            if not check_memory_available(mesh_memory):
                # Create recoverable error with checkpoint data
                from .utils.errors import RecoverableError

                raise RecoverableError(
                    "Insufficient memory for mesh generation",
                    f"Required: {mesh_memory:.1f}MB",
                    context={"volume_shape": volume_shape},
                    checkpoint_data={"stage": "ct_processing", "completed": True},
                )

            # Generate mesh
            if not checkpoint_data or checkpoint_data["stage"] in (
                "init",
                "ct_processing",
            ):
                mesh = self._generate_mesh()
                self.checkpoint_manager.save_checkpoint(
                    "mesh_generation", {"completed": True}
                )

            # Calculate material properties
            if not checkpoint_data or checkpoint_data["stage"] in (
                "init",
                "ct_processing",
                "mesh_generation",
            ):
                materials = self._assign_materials(mesh)
                self.checkpoint_manager.save_checkpoint(
                    "material_assignment", {"completed": True}
                )

            # Export results
            if not checkpoint_data or checkpoint_data["stage"] in (
                "init",
                "ct_processing",
                "mesh_generation",
                "material_assignment",
            ):
                self._export_results(mesh, materials)
                self.checkpoint_manager.save_checkpoint("export", {"completed": True})

            # Generate reports
            self._generate_reports()

            elapsed_time = time.time() - start_time
            self.logger.info(f"Pipeline completed in {elapsed_time:.1f} seconds")

            # Stop profiling and generate performance report
            self.monitor.stop_profiling()
            self.monitor.generate_report()

            # Clear checkpoint after successful completion
            self.checkpoint_manager.clear_checkpoint()

        except RecoverableError as e:
            self.logger.error(f"Pipeline failed with recoverable error: {str(e)}")
            # Save checkpoint for later resumption
            e.save_checkpoint(self.output_dir)
            raise

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            # Create emergency checkpoint with current state
            context = {
                "error": str(e),
                "stage": self.stats.get("current_stage", "unknown"),
            }
            self.checkpoint_manager.save_checkpoint("error", context)
            raise

    @profile_stage()
    def _get_ct_metadata(self) -> Dict[str, Any]:
        """Get CT metadata without loading full volume"""
        self.logger.start_stage("CT Metadata Analysis")
        self.stats["current_stage"] = "ct_metadata"

        # Get metadata about CT stack
        metadata = get_ct_metadata(Path(self.config.input_folder))
        self.logger.info(
            f"CT stack contains {metadata['n_slices']} slices, "
            f"dimensions: {metadata['height']}x{metadata['width']}, "
            f"estimated size: {metadata['total_size_mb']:.1f}MB"
        )

        # Generate validation reports
        self.validation.save_metrics()
        self.validation.generate_report()

        self.logger.end_stage()
        return metadata

    @profile_stage()
    def _process_ct_streaming(self) -> None:
        """Process CT volume data in chunks to reduce memory usage"""
        self.logger.start_stage("CT Processing (Streaming)")
        self.stats["current_stage"] = "ct_processing"

        input_folder = Path(self.config.input_folder)
        tiff_files = sorted(input_folder.glob("*.tif*"))

        # Calculate global percentiles for normalization
        # This requires a first pass through the data
        self.logger.info("Calculating global intensity statistics")
        min_vals = []
        max_vals = []

        # Process in chunks to find global min/max
        for chunk in create_ct_iterator(tiff_files, self.chunk_size):
            min_vals.append(np.percentile(chunk, self.config.clip_percentiles[0]))
            max_vals.append(np.percentile(chunk, self.config.clip_percentiles[1]))

        # Calculate global percentiles
        global_min = np.min(min_vals)
        global_max = np.max(max_vals)
        self.stats["normalization"] = {
            "clip_min": float(global_min),
            "clip_max": float(global_max),
        }

        self.logger.info(
            f"Global intensity range: [{global_min:.1f}, {global_max:.1f}]"
        )

        # Process chunks and save directly
        self.logger.info("Processing CT data in chunks")

        # Create a generator for processed chunks
        def process_chunks():
            for i, chunk in enumerate(create_ct_iterator(tiff_files, self.chunk_size)):
                self.logger.info(f"Processing chunk {i + 1}")

                # Normalize chunk
                chunk_norm = ((chunk - global_min) / (global_max - global_min)).astype(
                    np.float32
                )
                chunk_norm = np.clip(chunk_norm, 0, 1)

                # Optional coarsening
                if self.config.coarsen_factor > 1:
                    chunk_norm = coarsen_volume(chunk_norm, self.config.coarsen_factor)

                # Denoising if enabled
                if self.config.denoise_method:
                    chunk_norm = denoise_volume(
                        chunk_norm, self.config.denoise_method, self.config
                    )

                # Visualize first chunk
                if i == 0 and self.config.save_visualization:
                    visualization.visualize_slice(
                        chunk_norm,
                        self.output_dir,
                        self.config,
                        name=f"chunk_{i}_processed",
                    )

                yield chunk_norm

        # Save processed chunks
        save_ct_chunks(self.output_dir, process_chunks(), self.config)

        self.logger.end_stage()

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
