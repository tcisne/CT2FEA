# ct2fea/main.py
from pathlib import Path
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
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
from .utils.validation import (
    validate_output_path,
    validate_mesh,
    validate_material_properties,
)

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_ct_data(
    input_folder: str, output_dir: str, config: Config
) -> Dict[str, Any]:
    """Process CT data from TIFF files

    Args:
        input_folder: Path to folder containing TIFF files
        output_dir: Path to output directory
        config: Configuration object

    Returns:
        Dictionary with processing results and metadata
    """
    logger.info("Starting CT data processing")
    start_time = time.time()

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get metadata about CT stack
    input_path = Path(input_folder)
    metadata = get_ct_metadata(input_path)
    logger.info(
        f"CT stack contains {metadata['n_slices']} slices, "
        f"dimensions: {metadata['height']}x{metadata['width']}, "
        f"estimated size: {metadata['total_size_mb']:.1f}MB"
    )

    # Process CT data in chunks
    tiff_files = sorted(input_path.glob("*.tif*"))

    # Calculate global percentiles for normalization
    logger.info("Calculating global intensity statistics")
    min_vals = []
    max_vals = []

    # Process in chunks to find global min/max
    for chunk in create_ct_iterator(tiff_files, config.chunk_size):
        min_vals.append(np.percentile(chunk, config.clip_percentiles[0]))
        max_vals.append(np.percentile(chunk, config.clip_percentiles[1]))

    # Calculate global percentiles
    global_min = np.min(min_vals)
    global_max = np.max(max_vals)

    logger.info(f"Global intensity range: [{global_min:.1f}, {global_max:.1f}]")

    # Process chunks and save directly
    logger.info("Processing CT data in chunks")

    # Create a generator for processed chunks
    def process_chunks():
        for i, chunk in enumerate(create_ct_iterator(tiff_files, config.chunk_size)):
            logger.info(f"Processing chunk {i + 1}")

            # Normalize chunk
            chunk_norm = ((chunk - global_min) / (global_max - global_min)).astype(
                np.float32
            )
            chunk_norm = np.clip(chunk_norm, 0, 1)

            # Optional coarsening
            if config.coarsen_factor > 1:
                chunk_norm = coarsen_volume(chunk_norm, config.coarsen_factor)

            # Denoising if enabled
            if config.denoise_method:
                chunk_norm = denoise_volume(chunk_norm, config.denoise_method, config)

            # Save visualization of first chunk
            if i == 0 and config.save_visualization:
                from .visualization import visualize_slice

                visualize_slice(chunk_norm, output_path, name="ct_slice_sample")

            yield chunk_norm

    # Save processed chunks
    processed_path = save_ct_chunks(output_path, process_chunks(), config)

    elapsed_time = time.time() - start_time
    logger.info(f"CT processing completed in {elapsed_time:.1f} seconds")

    return {
        "metadata": metadata,
        "processed_path": processed_path,
        "normalization": {
            "min": float(global_min),
            "max": float(global_max),
        },
    }


def segment_ct_data(
    ct_data: np.ndarray, output_dir: str, config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment CT data into bone and pore regions

    Args:
        ct_data: Processed CT data as numpy array
        output_dir: Path to output directory
        config: Configuration object

    Returns:
        Tuple of (bone_mask, pore_mask)
    """
    logger.info("Starting CT segmentation")
    start_time = time.time()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Perform segmentation
    bone_mask, pore_mask = segment_volume(ct_data, config)

    # Validate segmentation
    validation_metrics = validate_segmentation(bone_mask, pore_mask)
    logger.info(
        f"Segmentation complete - Bone fraction: {validation_metrics['bone_fraction']:.3f}, "
        f"Pore fraction: {validation_metrics['pore_fraction']:.3f}"
    )

    # Save visualization if enabled
    if config.save_visualization:
        from .visualization import visualize_segmentation

        visualize_segmentation(ct_data, bone_mask, pore_mask, output_path)

    elapsed_time = time.time() - start_time
    logger.info(f"Segmentation completed in {elapsed_time:.1f} seconds")

    return bone_mask, pore_mask


def generate_mesh(
    bone_mask: np.ndarray, pore_mask: np.ndarray, output_dir: str, config: Config
) -> Tuple[pv.UnstructuredGrid, Dict]:
    """Generate mesh from segmentation masks

    Args:
        bone_mask: Binary mask of bone regions
        pore_mask: Binary mask of pore regions
        output_dir: Path to output directory
        config: Configuration object

    Returns:
        Tuple of (mesh, material_info)
    """
    logger.info("Starting mesh generation")
    start_time = time.time()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create mesh
    mesh, material_info = create_voxel_mesh(bone_mask, pore_mask, config)

    # Validate mesh
    validate_mesh(mesh)

    logger.info(
        f"Generated mesh with {mesh.n_cells} elements and {mesh.n_points} nodes"
    )
    logger.info(f"Material distribution: {material_info}")

    # Save visualization if enabled
    if config.save_visualization:
        from .visualization import visualize_mesh

        visualize_mesh(mesh, output_path)

    elapsed_time = time.time() - start_time
    logger.info(f"Mesh generation completed in {elapsed_time:.1f} seconds")

    return mesh, material_info


def assign_materials(
    ct_data: np.ndarray, mesh: pv.UnstructuredGrid, output_dir: str, config: Config
) -> Dict:
    """Assign material properties to mesh

    Args:
        ct_data: Processed CT data
        mesh: PyVista mesh
        output_dir: Path to output directory
        config: Configuration object

    Returns:
        Dictionary with material properties
    """
    logger.info("Starting material property assignment")
    start_time = time.time()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate material properties
    mapper = get_material_mapper(config)
    materials = mapper.calculate_properties(ct_data)

    # Validate properties
    validate_material_properties(materials, config.material_model)

    # Save visualization if enabled
    if config.save_visualization:
        from .visualization import visualize_materials

        visualize_materials(mesh, materials, output_path)

    elapsed_time = time.time() - start_time
    logger.info(f"Material assignment completed in {elapsed_time:.1f} seconds")

    return materials


def export_model(
    mesh: pv.UnstructuredGrid, materials: Dict, output_dir: str, config: Config
) -> str:
    """Export mesh and materials to FEA format

    Args:
        mesh: PyVista mesh
        materials: Material properties
        output_dir: Path to output directory
        config: Configuration object

    Returns:
        Path to exported file
    """
    logger.info("Starting model export")
    start_time = time.time()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export to Abaqus format
    output_file = output_path / "model.inp"
    export_to_abaqus(mesh, materials, config, output_file)

    elapsed_time = time.time() - start_time
    logger.info(f"Model export completed in {elapsed_time:.1f} seconds")

    return str(output_file)


def run_pipeline(config: Config) -> Dict[str, Any]:
    """Run the complete CT2FEA pipeline

    Args:
        config: Configuration object

    Returns:
        Dictionary with results
    """
    logger.info("Starting CT2FEA pipeline")
    start_time = time.time()

    # Validate configuration
    config.validate()

    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process CT data
    ct_results = process_ct_data(config.input_folder, config.output_dir, config)

    # Load processed CT data
    ct_data = np.load(ct_results["processed_path"])

    # Segment CT data
    bone_mask, pore_mask = segment_ct_data(ct_data, config.output_dir, config)

    # Generate mesh
    mesh, material_info = generate_mesh(bone_mask, pore_mask, config.output_dir, config)

    # Assign materials
    materials = assign_materials(ct_data, mesh, config.output_dir, config)

    # Export model
    output_file = export_model(mesh, materials, config.output_dir, config)

    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.1f} seconds")

    return {
        "ct_results": ct_results,
        "mesh_info": material_info,
        "output_file": output_file,
    }


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

    run_pipeline(config)
