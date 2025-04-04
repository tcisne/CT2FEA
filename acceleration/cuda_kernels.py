from numba import cuda
import numpy as np


@cuda.jit(device=True)
def dev_interpolate(coord, grid, spacing):
    """CUDA device function for 3D trilinear interpolation

    Args:
        coord: 3D coordinate to interpolate at
        grid: CT volume data
        spacing: Grid spacing
    Returns:
        Interpolated value
    """
    # Calculate base indices
    i0 = int(coord[0] / spacing[0])
    j0 = int(coord[1] / spacing[1])
    k0 = int(coord[2] / spacing[2])

    # Ensure we don't go out of bounds
    if (
        i0 < 0
        or i0 >= grid.shape[0] - 1
        or j0 < 0
        or j0 >= grid.shape[1] - 1
        or k0 < 0
        or k0 >= grid.shape[2] - 1
    ):
        return 0.0

    # Calculate fractional components
    xd = (coord[0] / spacing[0]) - i0
    yd = (coord[1] / spacing[1]) - j0
    zd = (coord[2] / spacing[2]) - k0

    # Trilinear interpolation
    c000 = grid[i0, j0, k0]
    c001 = grid[i0, j0, k0 + 1]
    c010 = grid[i0, j0 + 1, k0]
    c011 = grid[i0, j0 + 1, k0 + 1]
    c100 = grid[i0 + 1, j0, k0]
    c101 = grid[i0 + 1, j0, k0 + 1]
    c110 = grid[i0 + 1, j0 + 1, k0]
    c111 = grid[i0 + 1, j0 + 1, k0 + 1]

    return (
        c000 * (1 - xd) * (1 - yd) * (1 - zd)
        + c001 * (1 - xd) * (1 - yd) * zd
        + c010 * (1 - xd) * yd * (1 - zd)
        + c011 * (1 - xd) * yd * zd
        + c100 * xd * (1 - yd) * (1 - zd)
        + c101 * xd * (1 - yd) * zd
        + c110 * xd * yd * (1 - zd)
        + c111 * xd * yd * zd
    )


@cuda.jit
def kernel_material_properties(ct_volume, centroids, spacing, out_props):
    """Main CUDA kernel for material property calculation

    Args:
        ct_volume: Input CT volume data
        centroids: Element centroid coordinates
        spacing: Grid spacing
        out_props: Output material properties array
    """
    idx = cuda.grid(1)
    if idx < centroids.shape[0]:
        # Get interpolated CT value
        ct_val = dev_interpolate(centroids[idx], ct_volume, spacing)

        # Calculate material properties
        density = 1.0 + ct_val
        E = 10000.0 * (density * density)  # Example relationship

        # Store results
        out_props[idx, 0] = density
        out_props[idx, 1] = E
        out_props[idx, 2] = 0.3  # Poisson's ratio
