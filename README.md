# CT2FEA: CT to FEA Conversion Pipeline

A high-performance tool for converting CT (Computed Tomography) data to Finite Element Analysis (FEA) models, with support for GPU acceleration.

## Features

- Load and process CT image stacks (TIFF format)
- Advanced denoising with multiple algorithms (Gaussian, NLM, TV)
- Automatic bone and pore segmentation
- High-quality hexahedral mesh generation
- Multiple material models (linear elastic, plasticity, hyperelastic)
- GPU acceleration support (CUDA and OpenCL)
- Comprehensive mesh quality analysis
- Export to Abaqus INP format
- Interactive visualization and reporting

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

4. (Optional) Install GPU dependencies:
- For CUDA support: `pip install numba`
- For OpenCL support: `pip install pyopencl`

## Usage

### GUI Mode

Run the application in GUI mode:
```bash
python -m ct2fea
```

### Command Line Mode

Process a CT stack with default settings:
```bash
python -m ct2fea --input /path/to/ct/stack --output /path/to/output
```

Available options:
- `--voxel-size`: Voxel size in micrometers (default: 10)
- `--material-model`: Material model type (linear/plasticity/hyperelastic)
- `--use-gpu`: Enable GPU acceleration
- `--denoise-method`: Denoising algorithm (gaussian/nlm/tv)
- `--coarsen-factor`: Volume coarsening factor

## Configuration

The application can be configured through:
1. GUI interface
2. Command-line arguments
3. Configuration file (config.json)

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
    "denoise_params": {
        "method": "nlm",
        "patch_size": 5
    }
}
```

## Pipeline Steps

1. **CT Data Loading**
   - Support for TIFF stacks
   - Automatic normalization and preprocessing

2. **Image Processing**
   - Multiple denoising options
   - Optional volume coarsening
   - Automatic parameter estimation

3. **Segmentation**
   - Automatic bone/pore detection
   - Morphological operations
   - Connected component analysis

4. **Mesh Generation**
   - Hexahedral element creation
   - Mesh quality checks
   - Optional smoothing

5. **Material Assignment**
   - Multiple material models
   - GPU-accelerated property calculation
   - Automatic device selection

6. **Export**
   - Abaqus INP format
   - Quality reports
   - Visualization outputs

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
pytest -m "not gpu"  # Skip GPU tests
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
- PyVista
- meshio
- CUDA/OpenCL (optional)

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
