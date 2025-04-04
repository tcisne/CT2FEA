import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
import pyvista as pv


def create_sphere(center, radius, shape):
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (
        (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    ) <= radius**2


def create_cylinder(p1, p2, radius, shape):
    """Create a cylinder from point p1 to p2 with given radius"""
    # Create coordinate grids
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # Convert to arrays that match the grid shape
    z = z - p1[0]
    y = y - p1[1]
    x = x - p1[2]

    # Vector in direction of cylinder
    v = np.array(p2) - np.array(p1)
    length = np.sqrt(np.sum(v**2))
    v = v / length

    # Project points onto cylinder axis
    dot_product = z * v[0] + y * v[1] + x * v[2]

    # Calculate distance from points to cylinder axis
    dist_squared = (z**2 + y**2 + x**2) - (dot_product**2)

    # Create cylinder mask
    mask = (dist_squared <= radius**2) & (dot_product >= 0) & (dot_product <= length)

    return mask


def generate_am_metal(
    dimensions=(50, 50, 50),
    base_intensity=150,
    noise_level=20,
    n_pores=20,  # More pores
    min_radius=3.0,  # Larger pores
    max_radius=4.0,
):
    """Generate synthetic CT data similar to AM metal with spherical pores (~1-2% porosity)"""
    # Create binary mask for pores first
    pore_mask = np.zeros(dimensions, dtype=bool)

    # Add random spherical pores
    for _ in range(n_pores):
        center = np.random.randint(5, np.array(dimensions) - 5, size=3)
        radius = np.random.uniform(min_radius, max_radius)
        sphere_mask = create_sphere(center, radius, dimensions)
        pore_mask = pore_mask | sphere_mask

    # Create base volume with noise
    volume = np.full(dimensions, base_intensity, dtype=np.uint8)
    noise = np.random.randint(-noise_level, noise_level + 1, size=dimensions)
    volume = np.clip(volume + noise, 0, 255)

    # Apply pores
    volume[pore_mask] = 0
    return volume.astype(np.uint8)


def generate_cortical_bone(
    dimensions=(50, 50, 50),
    base_intensity=150,
    noise_level=20,
    n_tubules=15,  # Number of tubules
    tubule_radius=1.8,  # Radius
    max_angle=0.05,  # Very small maximum angle variation from Z-axis
):
    """Generate synthetic CT data similar to cortical bone with tubules (~1-2% porosity)
    running primarily in the Z-direction through the entire volume length"""
    # Create binary mask for tubules
    tubule_mask = np.zeros(dimensions, dtype=bool)

    # Base volume with noise
    volume = np.full(dimensions, base_intensity, dtype=np.uint8)
    noise = np.random.randint(-noise_level, noise_level + 1, size=dimensions)
    volume = np.clip(volume + noise, 0, 255)

    # Add tubules (aligned with z-axis with minimal variations)
    for _ in range(n_tubules):
        # Random position in xy plane - ensure tubules are not too close to edges
        x = np.random.randint(10, dimensions[0] - 10)
        y = np.random.randint(10, dimensions[1] - 10)

        # Very small random angle variation
        angle_x = np.random.uniform(-max_angle, max_angle)
        angle_y = np.random.uniform(-max_angle, max_angle)

        # Calculate end point with slight deviation
        end_x = x + int(dimensions[2] * np.tan(angle_x))
        end_y = y + int(dimensions[2] * np.tan(angle_y))

        # Ensure end points stay within bounds
        end_x = np.clip(end_x, 5, dimensions[0] - 5)
        end_y = np.clip(end_y, 5, dimensions[1] - 5)

        # Create tubule through entire Z length
        p1 = (0, y, x)  # Start at bottom
        p2 = (dimensions[2] - 1, end_y, end_x)  # End at top

        # Create tubule
        current_tubule = create_cylinder(p1, p2, tubule_radius, dimensions)
        tubule_mask = tubule_mask | current_tubule

    # Apply tubules
    volume[tubule_mask] = 0
    return volume.astype(np.uint8)


def visualize_3d(volume, threshold=10, title="3D Visualization"):
    """Create 3D visualization of the volume"""
    try:
        # Create PyVista grid
        grid = pv.ImageData()
        grid.dimensions = np.array(volume.shape) + 1
        grid.spacing = (1, 1, 1)
        grid.cell_data["values"] = volume.flatten(order="F")

        # Create material meshes
        material = grid.threshold([threshold, 255])
        pores = grid.threshold([0, threshold - 1])

        # Create plotter with better defaults
        p = pv.Plotter(window_size=[1024, 768])

        # Add materials with different colors and opacity
        p.add_mesh(material, color="wheat", opacity=0.7, name="Material")
        p.add_mesh(pores, color="blue", opacity=1.0, name="Pores")

        # Set camera position for better view
        p.camera_position = [
            (
                volume.shape[0] * 2,
                -volume.shape[1] * 1.5,
                volume.shape[2] * 2,
            ),  # Camera position
            (
                volume.shape[0] / 2,
                volume.shape[1] / 2,
                volume.shape[2] / 2,
            ),  # Focal point
            (0, 0, 1),  # Up vector
        ]

        # Add axes and title
        p.add_axes()
        p.add_title(title, font_size=16)

        # Show the plot
        p.show()

    except Exception as e:
        print(f"Error in 3D visualization: {str(e)}")
        print("Try updating pyvista with: pip install --upgrade pyvista")


# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(output_dir, exist_ok=True)

# Generate both types of synthetic data
dimensions = (50, 50, 50)
am_volume = generate_am_metal(dimensions)
bone_volume = generate_cortical_bone(dimensions)

# Save volumes
tifffile.imwrite(os.path.join(output_dir, "am_metal.tif"), am_volume)
tifffile.imwrite(os.path.join(output_dir, "cortical_bone.tif"), bone_volume)

# Visualize 2D slices
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# AM Metal visualization
axes[0, 0].imshow(am_volume[dimensions[0] // 2, :, :], cmap="gray")
axes[0, 0].set_title("AM Metal - XY Slice")
axes[0, 1].imshow(am_volume[:, dimensions[1] // 2, :], cmap="gray")
axes[0, 1].set_title("AM Metal - XZ Slice")

# Cortical Bone visualization
axes[1, 0].imshow(bone_volume[dimensions[0] // 2, :, :], cmap="gray")
axes[1, 0].set_title("Cortical Bone - XY Slice")
axes[1, 1].imshow(bone_volume[:, dimensions[1] // 2, :], cmap="gray")
axes[1, 1].set_title("Cortical Bone - XZ Slice")

plt.tight_layout()
plt.show()

# Show 3D visualizations
visualize_3d(am_volume, title="AM Metal 3D View")
visualize_3d(bone_volume, title="Cortical Bone 3D View")


# Print statistics
def print_stats(volume, name):
    print(f"\n{name} Statistics:")
    print(f"Dimensions: {volume.shape}")
    print(f"Intensity range: [{volume.min()}, {volume.max()}]")
    print(f"Mean intensity: {volume.mean():.1f}")
    porosity = (volume == 0).sum() / volume.size * 100
    print(f"Porosity: {porosity:.1f}%")


print_stats(am_volume, "AM Metal")
print_stats(bone_volume, "Cortical Bone")
