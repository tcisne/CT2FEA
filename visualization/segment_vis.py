# ct2fea/visualization/segment_vis.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import matplotlib.colors as mcolors


def visualize_segmentation(
    ct_data: np.ndarray,
    bone_mask: np.ndarray,
    pore_mask: np.ndarray,
    output_dir: Path,
    slice_idx: Optional[int] = None,
) -> str:
    """Visualize segmentation results

    Args:
        ct_data: Original CT data
        bone_mask: Binary mask of bone regions
        pore_mask: Binary mask of pore regions
        output_dir: Output directory
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
    ct_slice = ct_data[slice_idx]
    bone_slice = bone_mask[slice_idx]
    pore_slice = pore_mask[slice_idx]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot original CT slice
    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title("Original CT")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # Plot bone mask
    axes[1].imshow(bone_slice, cmap="bone")
    axes[1].set_title("Bone Segmentation")
    axes[1].set_xlabel("X")

    # Plot pore mask
    axes[2].imshow(pore_slice, cmap="cool")
    axes[2].set_title("Pore Segmentation")
    axes[2].set_xlabel("X")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = output_dir / "segmentation_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Create overlay visualization
    create_segmentation_overlay(ct_slice, bone_slice, pore_slice, output_dir)

    return str(output_path)


def create_segmentation_overlay(
    ct_slice: np.ndarray,
    bone_slice: np.ndarray,
    pore_slice: np.ndarray,
    output_dir: Path,
) -> str:
    """Create overlay of segmentation on CT slice

    Args:
        ct_slice: CT slice data
        bone_slice: Bone mask slice
        pore_slice: Pore mask slice
        output_dir: Output directory

    Returns:
        Path to saved image
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display CT slice as grayscale background
    ax.imshow(ct_slice, cmap="gray")

    # Create overlay mask
    overlay = np.zeros((*ct_slice.shape, 4))

    # Add bone regions in red (semi-transparent)
    overlay[bone_slice, :] = [1, 0, 0, 0.5]  # Red with 50% opacity

    # Add pore regions in blue (semi-transparent)
    overlay[pore_slice, :] = [0, 0, 1, 0.5]  # Blue with 50% opacity

    # Display overlay
    ax.imshow(overlay)

    # Add title and labels
    ax.set_title("Segmentation Overlay")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Add legend
    bone_patch = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.5)
    pore_patch = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5)
    ax.legend([bone_patch, pore_patch], ["Bone", "Pore"], loc="upper right")

    # Save figure
    output_path = output_dir / "segmentation_overlay.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)
