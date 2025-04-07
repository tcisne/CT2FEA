#!/usr/bin/env python3
"""
Example script demonstrating how to use the CT2FEA pipeline programmatically.
This example shows how to run the full pipeline and access the results.
"""

import os
import numpy as np
from pathlib import Path
from ct2fea.config import Config
from ct2fea.main import (
    process_ct_data,
    segment_ct_data,
    generate_mesh,
    assign_materials,
    export_model,
    run_pipeline,
)

# Define input and output paths
input_folder = "tests/test_data"  # Path to folder containing TIFF files
output_dir = "examples/output"  # Path to output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create configuration
config = Config(
    input_folder=input_folder,
    output_dir=output_dir,
    voxel_size_um=10.0,
    coarsen_factor=1,
    denoise_method="gaussian",
    pore_threshold=0.2,
    material_model="linear",
    material_params={
        "min_density": 1.0,
        "max_density": 2.0,
        "E_coeff": (10000, 2.0),
    },
    save_visualization=True,
)

# Run the full pipeline
print("Running full pipeline...")
results = run_pipeline(config)
print(f"Pipeline completed. Output file: {results['output_file']}")

# Alternatively, you can run each stage individually:
print("\nRunning individual stages...")

# 1. Process CT data
print("1. Processing CT data...")
ct_results = process_ct_data(input_folder, output_dir, config)
processed_ct_path = ct_results["processed_path"]
print(f"   CT data processed. Output: {processed_ct_path}")

# 2. Load processed CT data
print("2. Loading processed CT data...")
ct_data = np.load(processed_ct_path)
print(f"   CT data loaded. Shape: {ct_data.shape}")

# 3. Segment CT data
print("3. Segmenting CT data...")
bone_mask, pore_mask = segment_ct_data(ct_data, output_dir, config)
print(
    f"   Segmentation complete. Bone fraction: {np.mean(bone_mask):.3f}, Pore fraction: {np.mean(pore_mask):.3f}"
)

# 4. Generate mesh
print("4. Generating mesh...")
mesh, material_info = generate_mesh(bone_mask, pore_mask, output_dir, config)
print(f"   Mesh generated. Elements: {mesh.n_cells}, Nodes: {mesh.n_points}")

# 5. Assign materials
print("5. Assigning materials...")
materials = assign_materials(ct_data, mesh, output_dir, config)
print(f"   Materials assigned. Properties: {list(materials.keys())}")

# 6. Export model
print("6. Exporting model...")
output_file = export_model(mesh, materials, output_dir, config)
print(f"   Model exported. Output file: {output_file}")

print("\nAll stages completed successfully!")
