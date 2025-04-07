# ct2fea/cli.py
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Sequence
import json
import numpy as np
import pyvista as pv

from .config import Config
from .main import (
    process_ct_data,
    segment_ct_data,
    generate_mesh,
    assign_materials,
    export_model,
    run_pipeline,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments

    Args:
        args: Optional list of arguments, defaults to sys.argv[1:]

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="CT2FEA - CT to FEA Conversion Pipeline"
    )

    # Common options
    parser.add_argument("-o", "--output", type=str, help="Output directory")
    parser.add_argument("--config", type=str, help="JSON configuration file")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")

    # Create subparsers for different entry points
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Full pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input CT stack folder"
    )
    add_ct_options(pipeline_parser)
    add_segmentation_options(pipeline_parser)
    add_mesh_options(pipeline_parser)
    add_material_options(pipeline_parser)

    # CT processing only
    ct_parser = subparsers.add_parser("process-ct", help="Process CT data only")
    ct_parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input CT stack folder"
    )
    add_ct_options(ct_parser)

    # Segmentation only
    segment_parser = subparsers.add_parser("segment", help="Segment CT data")
    segment_parser.add_argument("-i", "--input", type=str, help="Input CT stack folder")
    segment_parser.add_argument(
        "--processed-ct", type=str, help="Pre-processed CT data file"
    )
    add_segmentation_options(segment_parser)

    # Mesh generation only
    mesh_parser = subparsers.add_parser(
        "generate-mesh", help="Generate mesh from segmentation"
    )
    mesh_parser.add_argument(
        "--bone-mask", type=str, help="Bone segmentation mask file"
    )
    mesh_parser.add_argument(
        "--pore-mask", type=str, help="Pore segmentation mask file"
    )
    add_mesh_options(mesh_parser)

    # Material assignment only
    material_parser = subparsers.add_parser(
        "assign-materials", help="Assign material properties"
    )
    material_parser.add_argument("--mesh", type=str, help="Mesh file")
    material_parser.add_argument("--ct-data", type=str, help="Processed CT data file")
    add_material_options(material_parser)

    # Export only
    export_parser = subparsers.add_parser("export", help="Export to FEA format")
    export_parser.add_argument("--mesh", type=str, required=True, help="Mesh file")
    export_parser.add_argument(
        "--materials", type=str, required=True, help="Material properties file"
    )

    # GUI mode
    parser.add_argument("--gui", action="store_true", help="Start in GUI mode")

    return parser.parse_args(args)


def add_ct_options(parser: argparse.ArgumentParser) -> None:
    """Add CT processing options to parser"""
    ct_group = parser.add_argument_group("CT Processing Options")
    ct_group.add_argument(
        "--voxel-size",
        type=float,
        default=10.0,
        help="Voxel size in micrometers (default: 10.0)",
    )
    ct_group.add_argument(
        "--coarsen-factor",
        type=int,
        default=1,
        help="Volume coarsening factor (default: 1)",
    )
    ct_group.add_argument(
        "--denoise-method",
        choices=["gaussian", "nlm", None],
        default=None,
        help="Denoising method to apply",
    )


def add_segmentation_options(parser: argparse.ArgumentParser) -> None:
    """Add segmentation options to parser"""
    seg_group = parser.add_argument_group("Segmentation Options")
    seg_group.add_argument(
        "--pore-threshold",
        type=float,
        default=0.2,
        help="Threshold for pore segmentation (default: 0.2)",
    )
    seg_group.add_argument(
        "--adaptive-threshold",
        action="store_true",
        help="Use adaptive thresholding for bone segmentation",
    )
    seg_group.add_argument(
        "--min-bone-size",
        type=int,
        default=100,
        help="Minimum bone region size in voxels (default: 100)",
    )
    seg_group.add_argument(
        "--min-pore-size",
        type=int,
        default=20,
        help="Minimum pore region size in voxels (default: 20)",
    )


def add_mesh_options(parser: argparse.ArgumentParser) -> None:
    """Add mesh generation options to parser"""
    mesh_group = parser.add_argument_group("Mesh Options")
    mesh_group.add_argument(
        "--mesh-format",
        choices=["inp", "vtk", "stl"],
        default="inp",
        help="Mesh output format (default: inp)",
    )


def add_material_options(parser: argparse.ArgumentParser) -> None:
    """Add material options to parser"""
    material_group = parser.add_argument_group("Material Options")
    material_group.add_argument(
        "--material-model",
        choices=["linear"],
        default="linear",
        help="Material model type (default: linear)",
    )
    material_group.add_argument(
        "--min-density",
        type=float,
        default=1.0,
        help="Minimum material density (default: 1.0)",
    )
    material_group.add_argument(
        "--max-density",
        type=float,
        default=2.0,
        help="Maximum material density (default: 2.0)",
    )


def load_config_file(path: Path) -> dict:
    """Load configuration from JSON file

    Args:
        path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        sys.exit(1)


def create_config_from_args(args: argparse.Namespace) -> Config:
    """Create configuration from command line arguments

    Args:
        args: Parsed command line arguments

    Returns:
        Configuration object
    """
    # Load from config file if specified
    if args.config:
        config_dict = load_config_file(Path(args.config))
    else:
        config_dict = {}

    # Set input and output paths
    if hasattr(args, "input") and args.input:
        config_dict["input_folder"] = args.input
    if args.output:
        config_dict["output_dir"] = args.output

    # CT processing options
    if hasattr(args, "voxel_size") and args.voxel_size:
        config_dict["voxel_size_um"] = args.voxel_size
    if hasattr(args, "coarsen_factor") and args.coarsen_factor:
        config_dict["coarsen_factor"] = args.coarsen_factor
    if hasattr(args, "denoise_method") and args.denoise_method:
        config_dict["denoise_method"] = args.denoise_method

    # Segmentation options
    if hasattr(args, "pore_threshold") and args.pore_threshold:
        config_dict["pore_threshold"] = args.pore_threshold

    segmentation_params = {}
    if hasattr(args, "adaptive_threshold") and args.adaptive_threshold:
        segmentation_params["use_adaptive_threshold"] = True
    if hasattr(args, "min_bone_size") and args.min_bone_size:
        segmentation_params["min_bone_size"] = args.min_bone_size
    if hasattr(args, "min_pore_size") and args.min_pore_size:
        segmentation_params["min_pore_size"] = args.min_pore_size

    if segmentation_params:
        config_dict["segmentation_params"] = segmentation_params

    # Material options
    if hasattr(args, "material_model") and args.material_model:
        config_dict["material_model"] = args.material_model

    material_params = {}
    if hasattr(args, "min_density") and args.min_density:
        material_params["min_density"] = args.min_density
    if hasattr(args, "max_density") and args.max_density:
        material_params["max_density"] = args.max_density

    if material_params:
        config_dict["material_params"] = material_params

    # Visualization options
    if args.no_vis:
        config_dict["save_visualization"] = False

    # Create config object
    try:
        return Config(**config_dict)
    except Exception as e:
        logger.error(f"Error creating configuration: {e}")
        sys.exit(1)


def cli_main() -> None:
    """Main CLI entry point"""
    args = parse_args()

    # Use GUI mode if requested
    if args.gui:
        from .io.gui import get_gui_inputs

        config = get_gui_inputs()
        if not config:
            return
        run_pipeline(config)
        return

    # Create configuration
    config = create_config_from_args(args)

    # Run requested command
    if args.command == "pipeline" or args.command is None:
        # Run full pipeline
        run_pipeline(config)

    elif args.command == "process-ct":
        # Process CT data only
        process_ct_data(config.input_folder, config.output_dir, config)

    elif args.command == "segment":
        # Segment CT data
        if args.processed_ct:
            # Load pre-processed CT data
            ct_data = np.load(args.processed_ct)
        elif args.input:
            # Process CT data first
            results = process_ct_data(args.input, config.output_dir, config)
            ct_data = np.load(results["processed_path"])
        else:
            logger.error("Either --processed-ct or --input must be specified")
            sys.exit(1)

        segment_ct_data(ct_data, config.output_dir, config)

    elif args.command == "generate-mesh":
        # Generate mesh from segmentation masks
        if not args.bone_mask or not args.pore_mask:
            logger.error("Both --bone-mask and --pore-mask must be specified")
            sys.exit(1)

        # Load segmentation masks
        bone_mask = np.load(args.bone_mask)
        pore_mask = np.load(args.pore_mask)

        generate_mesh(bone_mask, pore_mask, config.output_dir, config)

    elif args.command == "assign-materials":
        # Assign material properties
        if not args.mesh or not args.ct_data:
            logger.error("Both --mesh and --ct-data must be specified")
            sys.exit(1)

        # Load mesh and CT data
        mesh = pv.read(args.mesh)
        ct_data = np.load(args.ct_data)

        assign_materials(ct_data, mesh, config.output_dir, config)

    elif args.command == "export":
        # Export to FEA format
        if not args.mesh or not args.materials:
            logger.error("Both --mesh and --materials must be specified")
            sys.exit(1)

        # Load mesh and materials
        mesh = pv.read(args.mesh)
        with open(args.materials, "r") as f:
            materials = json.load(f)

        export_model(mesh, materials, config.output_dir, config)
