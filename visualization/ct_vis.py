# ct2fea/visualization/ct_vis.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def visualize_slice(
    ct_data: np.ndarray,
    output_dir: Path,
    name: str = "ct_slice",
    slice_idx: Optional[int] = None,
) -> str:
    """Visualize a slice of CT data

    Args:
        ct_data: 3D CT data array
        output_dir: Output directory
        name: Base name for output file
        slice_idx: Index of slice to visualize (default: middle slice)

    Returns:
        Path to saved image
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select slice to visualize
    if slice_idx is None:
        slice_idx = ct_data.shape[0] // 2

    # Ensure slice index is valid
    slice_idx = max(0, min(slice_idx, ct_data.shape[0] - 1))

    # Get slice data
    if len(ct_data.shape) == 3:
        slice_data = ct_data[slice_idx]
    else:
        slice_data = ct_data  # Assume it's already a 2D slice

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display slice
    im = ax.imshow(slice_data, cmap="gray")

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Intensity")

    # Add title and labels
    ax.set_title(f"CT Slice {slice_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Save figure
    output_path = output_dir / f"{name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def visualize_histogram(
    ct_data: np.ndarray,
    output_dir: Path,
    name: str = "ct_histogram",
) -> str:
    """Visualize histogram of CT data

    Args:
        ct_data: CT data array
        output_dir: Output directory
        name: Base name for output file

    Returns:
        Path to saved image
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    ax.hist(ct_data.flatten(), bins=100, alpha=0.7)

    # Add title and labels
    ax.set_title("CT Data Histogram")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")

    # Save figure
    output_path = output_dir / f"{name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)
