#!/usr/bin/env python3
"""
Example script demonstrating how to use different entry points of the CT2FEA pipeline.
This example shows how to start from different stages of the pipeline.
"""

import os
import numpy as np
import pyvista as pv
import json
from pathlib import Path
from ct2fea.config import Config
from ct2fea.main import (
    process_ct_data,
    segment_ct_data,
    generate_mesh,
    assign_materials,
    export_model,
)

# Define paths
input_folder = "tests/test_data"  # Path to folder containing TIFF files
output_dir = "examples/output"  # Path to output directory
os.makedirs(output_dir, exist_ok=True)

# Create configuration
config = Config(
    input_folder=input_folder,
    output_dir=output_dir,
    voxel_size_um=10.0,
    denoise_method="gaussian",
    save_visualization=True,
)

# Example 1: Starting from raw CT data
print("Example 1: Starting from raw CT data")
print("------------------------------------")

# Process CT data
ct_results = process_ct_data(input_folder, output_dir, config)
processed_ct_path = ct_results["processed_path"]
print(f"CT data processed. Output: {processed_ct_path}")

# Example 2: Starting from pre-processed CT data
print("\nExample 2: Starting from pre-processed CT data")
print("---------------------------------------------")

# Load pre-processed CT data
ct_data = np.load(processed_ct_path)
print(f"Pre-processed CT data loaded. Shape: {ct_data.shape}")

# Segment CT data
bone_mask, pore_mask = segment_ct_data(ct_data, output_dir, config)
print(f"Segmentation complete.")

# Save segmentation masks for later use
bone_mask_path = os.path.join(output_dir, "bone_mask.npy")
pore_mask_path = os.path.join(output_dir, "pore_mask.npy")
np.save(bone_mask_path, bone_mask)
np.save(pore_mask_path, pore_mask)
print(f"Segmentation masks saved to {bone_mask_path} and {pore_mask_path}")

# Example 3: Starting from segmentation masks
print("\nExample 3: Starting from segmentation masks")
print("------------------------------------------")

# Load segmentation masks
bone_mask = np.load(bone_mask_path)
pore_mask = np.load(pore_mask_path)
print(f"Segmentation masks loaded. Shapes: {bone_mask.shape}, {pore_mask.shape}")

# Generate mesh
mesh, material_info = generate_mesh(bone_mask, pore_mask, output_dir, config)
print(f"Mesh generated. Elements: {mesh.n_cells}, Nodes: {mesh.n_points}")

# Save mesh for later use
mesh_path = os.path.join(output_dir, "mesh.vtk")
mesh.save(mesh_path)
print(f"Mesh saved to {mesh_path}")

# Example 4: Starting from existing mesh
print("\nExample 4: Starting from existing mesh")
print("-------------------------------------")

# Load mesh
mesh = pv.read(mesh_path)
print(f"Mesh loaded. Elements: {mesh.n_cells}, Nodes: {mesh.n_points}")

# Assign materials
materials = assign_materials(ct_data, mesh, output_dir, config)
print(f"Materials assigned. Properties: {list(materials.keys())}")

# Save materials for later use
materials_path = os.path.join(output_dir, "materials.json")
with open(materials_path, "w") as f:
    # Convert numpy arrays to lists for JSON serialization
    serializable_materials = {}
    for key, value in materials.items():
        if isinstance(value, np.ndarray):
            serializable_materials[key] = value.tolist()
        else:
            serializable_materials[key] = value
    json.dump(serializable_materials, f)
print(f"Materials saved to {materials_path}")

# Example 5: Starting from existing mesh and materials
print("\nExample 5: Starting from existing mesh and materials")
print("--------------------------------------------------")

# Load mesh and materials
mesh = pv.read(mesh_path)
with open(materials_path, "r") as f:
    materials_json = json.load(f)

# Convert lists back to numpy arrays
materials = {}
for key, value in materials_json.items():
    if isinstance(value, list):
        materials[key] = np.array(value)
    else:
        materials[key] = value

print(f"Mesh and materials loaded.")

# Export model
output_file = export_model(mesh, materials, output_dir, config)
print(f"Model exported. Output file: {output_file}")

print("\nAll examples completed successfully!")
