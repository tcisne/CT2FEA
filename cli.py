# ct2fea/cli.py
import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence
import json
from .config import Config
from .main import main


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

    # Input/Output options
    parser.add_argument("-i", "--input", type=str, help="Input CT stack folder")
    parser.add_argument("-o", "--output", type=str, help="Output directory")

    # Processing options
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=10.0,
        help="Voxel size in micrometers (default: 10.0)",
    )
    parser.add_argument(
        "--coarsen-factor",
        type=int,
        default=1,
        help="Volume coarsening factor (default: 1)",
    )
    parser.add_argument(
        "--denoise-method",
        choices=["gaussian", "nlm", "tv", None],
        default=None,
        help="Denoising method to apply",
    )

    # Material options
    parser.add_argument(
        "--material-model",
        choices=["linear", "plasticity", "hyperelastic"],
        default="linear",
        help="Material model type (default: linear)",
    )
    parser.add_argument(
        "--material-params", type=str, help="JSON file containing material parameters"
    )

    # Performance options
    performance_group = parser.add_argument_group("Performance Options")
    performance_group.add_argument(
        "--use-gpu", action="store_true", help="Enable GPU acceleration"
    )
    performance_group.add_argument(
        "--parallel", action="store_true", help="Enable parallel processing"
    )
    performance_group.add_argument(
        "--jobs", type=int, help="Number of parallel jobs (default: use all cores)"
    )

    # Visualization options
    vis_group = parser.add_argument_group("Visualization Options")
    vis_group.add_argument(
        "--interactive", action="store_true", help="Enable interactive 3D visualization"
    )
    vis_group.add_argument(
        "--no-vis", action="store_true", help="Disable all visualization"
    )

    # Configuration
    parser.add_argument("--config", type=str, help="JSON configuration file")

    # GUI mode
    parser.add_argument("--gui", action="store_true", help="Start in GUI mode")

    return parser.parse_args(args)


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
        print(f"Error loading configuration file: {e}", file=sys.stderr)
        sys.exit(1)


def create_config_from_args(args: argparse.Namespace) -> Optional[Config]:
    """Create configuration from command line arguments

    Args:
        args: Parsed command line arguments

    Returns:
        Configuration object or None if using GUI
    """
    # Use GUI mode if requested or no input/output specified
    if args.gui or (not args.input and not args.output and not args.config):
        return None

    # Load from config file if specified
    if args.config:
        config_dict = load_config_file(Path(args.config))
    else:
        config_dict = {}

    # Override with command line arguments
    if args.input:
        config_dict["input_folder"] = args.input
    if args.output:
        config_dict["output_dir"] = args.output
    if args.voxel_size:
        config_dict["voxel_size_um"] = args.voxel_size
    if args.coarsen_factor:
        config_dict["coarsen_factor"] = args.coarsen_factor
    if args.denoise_method:
        config_dict["denoise_method"] = args.denoise_method
    if args.material_model:
        config_dict["material_model"] = args.material_model
    if args.use_gpu:
        config_dict["use_gpu"] = True

    # Add parallel processing options
    if args.parallel:
        config_dict["use_parallel"] = True
        if args.jobs:
            config_dict["n_jobs"] = args.jobs

    # Add visualization options
    if args.no_vis:
        config_dict["visualization_dpi"] = 0
    elif args.interactive:
        config_dict["visualization_dpi"] = 300

    # Load material parameters if specified
    if args.material_params:
        material_params = load_config_file(Path(args.material_params))
        config_dict["material_params"] = material_params

    try:
        return Config(**config_dict)
    except Exception as e:
        print(f"Error creating configuration: {e}", file=sys.stderr)
        sys.exit(1)


def cli_main() -> None:
    """Main CLI entry point"""
    args = parse_args()
    config = create_config_from_args(args)
    main(config)
