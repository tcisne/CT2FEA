# CT2FEA: CT to FEA Conversion Pipeline

A streamlined tool for converting CT (Computed Tomography) data to Finite Element Analysis (FEA) models, designed for accessibility and ease of use.

## Features

- Load and process CT image stacks (TIFF format)
- Basic denoising with Gaussian and NLM algorithms
- Bone and pore segmentation
- Hexahedral mesh generation
- Linear elastic material model
- Export to Abaqus INP format
- Basic visualization for verification

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CT2FEA.git
cd CT2FEA
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

CT2FEA provides multiple entry points to the pipeline, allowing you to start from different stages depending on your needs.

### GUI Mode

Run the application in GUI mode:
```bash
python -m ct2fea --gui
```

### Full Pipeline

Process a CT stack through the entire pipeline:
```bash
python -m ct2fea pipeline --input /path/to/ct/stack --output /path/to/output
```

### Individual Pipeline Stages

CT2FEA allows you to run individual stages of the pipeline:

#### 1. Process CT Data Only
```bash
python -m ct2fea process-ct --input /path/to/ct/stack --output /path/to/output
```

#### 2. Segment Pre-processed CT Data
```bash
python -m ct2fea segment --processed-ct /path/to/processed_ct.npy --output /path/to/output
```

#### 3. Generate Mesh from Segmentation Masks
```bash
python -m ct2fea generate-mesh --bone-mask /path/to/bone_mask.npy --pore-mask /path/to/pore_mask.npy --output /path/to/output
```

#### 4. Assign Materials to Existing Mesh
```bash
python -m ct2fea assign-materials --mesh /path/to/mesh.vtk --ct-data /path/to/processed_ct.npy --output /path/to/output
```

#### 5. Export to FEA Format
```bash
python -m ct2fea export --mesh /path/to/mesh.vtk --materials /path/to/materials.json --output /path/to/output
```

## Common Options

- `--voxel-size`: Voxel size in micrometers (default: 10)
- `--denoise-method`: Denoising algorithm (gaussian/nlm)
- `--coarsen-factor`: Volume coarsening factor
- `--pore-threshold`: Threshold for pore segmentation (default: 0.2)
- `--no-vis`: Disable visualization
- `--config`: Path to JSON configuration file

## Configuration

The application can be configured through:
1. Command-line arguments
2. Configuration file (JSON format)
3. GUI interface

Example configuration file:
```json
{
    "voxel_size_um": 10.0,
    "material_model": "linear",
    "material_params": {
        "min_density": 1.0,
        "max_density": 2.0,
        "E_coeff": [10000, 2.0]
    },
    "denoise_method": "gaussian",
    "pore_threshold": 0.2
}
```

## Pipeline Steps

1. **CT Data Processing**
   - Support for TIFF stacks
   - Normalization and optional denoising
   - Optional volume coarsening

2. **Segmentation**
   - Bone/pore detection
   - Morphological operations
   - Connected component analysis

3. **Mesh Generation**
   - Hexahedral element creation
   - Mesh quality checks

4. **Material Assignment**
   - Linear elastic material model
   - Density-based property calculation

5. **Export**
   - Abaqus INP format
   - Basic visualization outputs

## Visualization

CT2FEA provides basic visualization capabilities to verify results at each stage:

- CT slice visualization
- Segmentation overlay visualization
- 3D mesh visualization
- Material property visualization

All visualizations are saved as PNG images in the output directory.

## Development

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/test_processing.py
pytest tests/test_meshing.py
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## Requirements

- Python 3.8+
- NumPy
- SciPy
- scikit-image
- matplotlib
- PyVista
- meshio
- tifffile

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{ct2fea2023,
  title={CT2FEA: CT to FEA Conversion Pipeline},
  author={Your Name},
  year={2023},
  url={https://github.com/yourusername/CT2FEA}
}
```

## Acknowledgments

- scikit-image for image processing algorithms
- PyVista for visualization
- meshio for mesh I/O operations
