# QGSW PyTorch

PyTorch implementation of multi-layer quasi-geostrophic (QG) and shallow-water (SW) models from the paper  [A Unified Formulation of Quasi-Geostrophic and Shallow Water Equations via Projection](https://doi.org/10.1029/2024MS004510).

If you use it, please cite:
```
@article{https://doi.org/10.1029/2024MS004510,
author = {Thiry, Louis and Li, Long and Mémin, Etienne and Roullet, Guillaume},
title = {A Unified Formulation of Quasi-Geostrophic and Shallow Water Equations via Projection},
journal = {Journal of Advances in Modeling Earth Systems},
volume = {16},
number = {10},
pages = {e2024MS004510},
keywords = {ocean models, shallow water, quasigeostrophic, projection},
doi = {https://doi.org/10.1029/2024MS004510},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2024MS004510},
eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024MS004510},
note = {e2024MS004510 2024MS004510},
abstract = {Abstract This paper introduces a unified model for layered rotating shallow-water (RSW) and quasi-geostrophic (QG) equations, based on the intrinsic relationship between these two sets of equations. We propose a novel formulation of the QG equations as a projection of the RSW equations. This formulation uses the same prognostic variables as RSW, namely velocity and layer thickness, thereby restoring the proximity of these two sets of equations. It provides direct access to the ageostrophic velocities embedded within the geostrophic velocities resolved by the QG equations. This approach facilitates the study of differences between QG and RSW using a consistent numerical discretization. We demonstrate the effectiveness of this formulation through examples including vortex shear instability, double-gyre circulation, and a simplified North Atlantic configuration.},
year = {2024}
}
```

**All examples now support command-line arguments with argparse for easy configuration!**

## Demo

Vortex shear instability comparing QG and SW solutions (Ro=0.1, Bu=1.0):

<p align="center">
  <video src="https://github.com/user/repo/assets/vortexshear_Ro0.1_Bu1.0_192x192.mp4" controls>
    Your browser does not support the video tag.
  </video>
</p>

> **Note**: If the video doesn't display above, you can view it directly: [assets/vortexshear_Ro0.1_Bu1.0_192x192.mp4](assets/vortexshear_Ro0.1_Bu1.0_192x192.mp4)

*Example output from `vortexshear.py` showing the evolution of vorticity fields in QG (left), SW (middle), and their difference (right).*

## Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Code Structure](#code-structure)
- [Quick Reference](#quick-reference)
- [Example Scripts](#example-scripts)
  - [Vortex Shear](#1-vortex-shear-vortexshearpy)
  - [Double Gyre](#2-double-gyre-doublegyrepy)
  - [North Atlantic](#3-north-atlantic-natlpy)
- [Output Files](#output-files)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)

---

## Installation

### Install uv

First, install uv if you haven't already:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install the package

Install the package using [uv](https://github.com/astral-sh/uv):

```bash
# Create a virtual environment
uv venv

# Install the package
uv pip install -e .

# Optional: Install additional dependencies for realistic configurations
uv pip install -e ".[realistic]"
```

---

## Requirements

### Core Dependencies

- `torch>=2.0.0`
- `numpy>=1.20.0`
- `matplotlib>=3.5.0`
- `setuptools>=61.0`

### Optional Dependencies (for realistic configurations)

- `scipy>=1.7.0`
- `netCDF4>=1.5.0`
- `scikit-image>=0.19.0`

### System Requirements (for video creation)

**ffmpeg** is required for creating MP4 videos:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Tested Hardware

Intel CPUs and NVIDIA GPUs (RTX A3000 Laptop, RTX 2080Ti, V100, A100).

---

## Code Structure

The source code is in the `src/` directory:

- **`finite_diff.py`**: Finite difference calculations
- **`reconstruction.py`**: Reconstruction/interpolation for finite volume quantities
- **`flux.py`**: Flow calculations for finite volume methods
- **`helmholtz.py`**: Classical Helmholtz equation solver ($\Delta f - \lambda f = r$) using pseudo-spectral methods
- **`helmholtz_multigrid.py`**: Generalized Helmholtz equation solver ($\nabla \cdot \left( c \nabla f \right) - \lambda f = r$) by multigrid method
- **`masks.py`**: Mask logic for non-rectangular geometries
- **`qg.py`**: Quasi-geostrophic solver
- **`sw.py`**: Shallow-water solver

---

## Quick Reference

### Setup

```bash
cd /path/to/pyQGSW
cd examples
```

### Basic Commands

```bash
# See all options for any script
uv run python <script>.py --help

# Run with default parameters
uv run python vortexshear.py
uv run python doublegyre.py
uv run python natl.py
```

### Common Options

All examples support these common command-line arguments:

| Option | Description | Example |
|--------|-------------|---------|
| `--nx NX --ny NY` | Grid resolution | `--nx 512 --ny 512` |
| `--device DEVICE` | cpu/cuda/auto (default: auto) | `--device cuda` |
| `--compile` | Enable torch.compile | `--compile` |
| `--no-plot` | Disable interactive display | `--no-plot` |
| `--save-images` | Save PNG frames | `--save-images` |
| `--save-video` | Create MP4 video | `--save-video --fps 20` |
| `--fps N` | Video frame rate | `--fps 15` |
| `--output-dir DIR` | Custom output directory | `--output-dir my_run` |
| `--freq-plot N` | Plot frequency (days) | `--freq-plot 10` |

### Quick Examples

```bash
# Create a video (no display)
uv run python doublegyre.py --no-plot --save-video --fps 15

# High-resolution GPU run
uv run python vortexshear.py --nx 512 --ny 512 --device cuda --compile

# Save images for later processing
uv run python natl.py --no-plot --save-images --freq-plot 5

# Parameter sweep
uv run python doublegyre.py --model qg --wind-mag 0.1 --slip-coef 0.5

# Long simulation with checkpoints
uv run python doublegyre.py --n-years 50 --freq-save 30 --save-after-years 10
```

---

## Example Scripts

All example scripts are in the `examples/` directory and support `--help`:

```bash
cd examples
uv run python vortexshear.py --help
uv run python doublegyre.py --help
uv run python natl.py --help
```

### 1. Vortex Shear (`vortexshear.py`)

Compares QG and SW solutions for vortex shear instability.

**See the [demo video](#demo) at the top of this README for example output.**

#### Basic Usage

```bash
# Run with default parameters (interactive plot)
uv run python vortexshear.py

# Custom Rossby and Burger numbers
uv run python vortexshear.py --Ro 0.05 --Bu 0.5

# Higher resolution
uv run python vortexshear.py --nx 256 --ny 256
```

#### Create Video

```bash
# Create video without interactive display
uv run python vortexshear.py --no-plot --save-video --fps 20

# Save images for custom processing
uv run python vortexshear.py --no-plot --save-images --freq-plot 50
```

#### GPU Acceleration

```bash
# Use GPU with compilation
uv run python vortexshear.py --device cuda --compile --nx 384 --ny 384
```

#### Key Parameters

- `--nx`, `--ny`: Grid resolution (default: 192×192)
- `--L`: Domain size in meters (default: 100000)
- `--Ro`: Rossby number (default: 0.1)
- `--Bu`: Burger number (default: 1.0)
- `--n-tau`: Number of tau periods to simulate (default: 8.0)
- `--flip-sign`: Flip sign of Coriolis parameter
- `--freq-plot`: Plot frequency in timesteps (default: 100)

---

### 2. Double Gyre (`doublegyre.py`)

Idealized double-gyre configuration with wind forcing.

#### Basic Usage

```bash
# Run QG model for 10 years (default)
uv run python doublegyre.py

# Run SW model for 5 years
uv run python doublegyre.py --model sw --n-years 5

# Quick test with lower resolution
uv run python doublegyre.py --nx 128 --ny 128 --n-years 1
```

#### Long Simulations

```bash
# Long simulation with data saving
uv run python doublegyre.py --n-years 50 --freq-save 30 --save-after-years 10

# With video output (plots every 15 days)
uv run python doublegyre.py --n-years 10 --no-plot --save-video --freq-plot 15
```

#### Custom Physical Parameters

```bash
# Modify wind forcing and slip condition
uv run python doublegyre.py --wind-mag 0.1 --slip-coef 0.5

# Custom Coriolis parameters
uv run python doublegyre.py --f0 1e-4 --beta 2e-11
```

#### Restart from Saved State

```bash
# Save state periodically
uv run python doublegyre.py --n-years 10 --freq-save 10

# Restart from year 5
uv run python doublegyre.py --start-file run_outputs/qg_256x256_dt4000_slip1.0/uvh_005y_000d.npz --n-years 10
```

#### Key Parameters

- `--model qg|sw`: Choose model type (default: qg)
- `--nx`, `--ny`: Grid resolution (default: 256×256)
- `--Lx`, `--Ly`: Domain size in meters (default: 5120km × 5120km)
- `--n-years`: Simulation duration (default: 10 years)
- `--dt`: Time step in seconds (default: 4000, auto-adjusted for SW)
- `--f0`: Mean Coriolis parameter (default: 9.375e-5 s⁻¹)
- `--beta`: Coriolis gradient (default: 1.754e-11 m⁻¹s⁻¹)
- `--wind-mag`: Wind stress magnitude in Pa (default: 0.08)
- `--slip-coef`: Slip coefficient, 0=no-slip, 1=free-slip (default: 1.0)
- `--freq-save`: Save state every N days (default: 15, 0 to disable)
- `--save-after-years`: Start saving after N years (default: 2)
- `--layer`: Which layer to plot, 0-indexed (default: 0)

---

### 3. North Atlantic (`natl.py`)

Realistic North Atlantic configuration with coastal geometries.

#### Prerequisites

Install realistic dependencies:

```bash
# Install Python dependencies
uv pip install -e ".[realistic]"

# Install ffmpeg (for video creation)
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg
```

#### Basic Usage

```bash
# Run with default parameters (downloads data automatically)
uv run python natl.py

# Quick test with lower resolution
uv run python natl.py --nx 512 --ny 256 --n-years 1
```

**Note**: The first run will download required data files (~100MB):
- Topography data: `data/northatlantic_topo.mat`
- Wind stress climatology: `data/windstress_HellermanRosenstein83.nc`

#### Create Video

```bash
# Create video of 5-year simulation
uv run python natl.py --n-years 5 --no-plot --save-video --fps 15 --freq-plot 5

# Save images every 2 days
uv run python natl.py --n-years 2 --no-plot --save-images --freq-plot 2
```

#### High-Resolution Simulation

```bash
# High-res run on GPU
uv run python natl.py --nx 2048 --ny 1024 --device cuda --compile --n-years 10
```

#### Custom Domain

```bash
# Different latitude range
uv run python natl.py --lat-min 15 --lat-max 55 --nx 1024 --ny 512
```

#### Data Management

```bash
# Use custom data directory
uv run python natl.py --data-dir /path/to/data

# Custom output directory
uv run python natl.py --output-dir my_natl_run --n-years 5
```

#### Key Parameters

- `--model qg|sw`: Choose model type (default: qg)
- `--nx`, `--ny`: Grid resolution (default: 1024×512)
- `--lat-min`, `--lat-max`: Latitude range (default: 9° to 48°)
- `--n-years`: Simulation duration (default: 10 years)
- `--dt`: Time step in seconds (default: 2000)
- `--f0`: Mean Coriolis parameter (default: 9.375e-5 s⁻¹)
- `--beta`: Coriolis gradient (default: 1.754e-11 m⁻¹s⁻¹)
- `--slip-coef`: Slip coefficient (default: 0.6)
- `--data-dir`: Directory for downloaded data (default: ./data)
- `--topo-url`: Custom topography data URL
- `--wind-url`: Custom wind stress data URL
- `--freq-save`: Save state every N days (default: 5)
- `--freq-plot`: Plot every N days (default: 5)
- `--layer`: Which layer to plot, 0-indexed (default: 0)

---

## Output Files

### Directory Structure

```
run_outputs/{experiment_name}/
├── images/                    # PNG frames (if --save-images)
│   ├── frame_000001.png
│   ├── frame_000002.png
│   └── ...
├── {experiment_name}.mp4      # Video (if --save-video)
├── uvh_YYYy_DDDd.npz          # State files (if freq-save > 0)
├── param.pth                  # Parameters (natl.py only)
└── mask_*.npy                 # Masks (natl.py only)
```

### Images

When using `--save-images`, PNG files are saved to:
```
{output_dir}/images/frame_NNNNNN.png
```

### Videos

When using `--save-video`, MP4 files are created:
```
{output_dir}/vortexshear_RoX_BuY.mp4    # vortexshear.py
{output_dir}/qg_doublegyre.mp4          # doublegyre.py
{output_dir}/qg_natl.mp4                # natl.py
```

### State Files

When using `--freq-save N`, state files are saved as:
```
{output_dir}/uvh_YYYy_DDDd.npz
```

Where `YYY` is the year and `DDD` is the day. Load them with:

```python
import numpy as np
data = np.load('uvh_005y_180d.npz')
u = data['u']  # velocity fields
v = data['v']
h = data['h']  # layer thickness
```

---

## Advanced Usage

### Batch Processing

Create videos for multiple parameter combinations:

```bash
#!/bin/bash
for ro in 0.05 0.1 0.2; do
    for bu in 0.5 1.0 2.0; do
        uv run python vortexshear.py --Ro $ro --Bu $bu \
            --no-plot --save-video \
            --output-dir "run_outputs/Ro${ro}_Bu${bu}"
    done
done
```

### Performance Optimization

For large simulations:

1. **Use GPU**: `--device cuda`
2. **Enable compilation**: `--compile` (first run will be slow)
3. **Disable plotting during run**: `--no-plot`
4. **Create video from saved images later**:
   ```bash
   ffmpeg -framerate 10 -pattern_type glob -i 'images/frame_*.png' \
          -c:v libx264 -pix_fmt yuv420p output.mp4
   ```

### Monitoring Long Runs

Use `nohup` for background execution:

```bash
nohup uv run python natl.py --n-years 50 --no-plot --save-video > run.log 2>&1 &
```

Monitor progress:
```bash
tail -f run.log
```

### Debugging

For testing new configurations quickly:

```bash
# Very short run with frequent output
uv run python doublegyre.py --nx 64 --ny 64 --n-years 0.1 --freq-plot 0.01
```

---

## Troubleshooting

### Out of Memory

**Problem**: Simulation crashes with out of memory error.

**Solutions**:
- Reduce grid resolution: `--nx 128 --ny 128`
- Use CPU instead of GPU: `--device cpu`
- Close other applications

### Video Creation Fails

**Problem**: Video creation fails with ffmpeg error.

**Solutions**:
```bash
# Check if ffmpeg is installed
ffmpeg -version

# Install if missing:
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### Slow Performance

**Problem**: Simulation runs too slowly.

**Solutions**:
- Enable compilation: `--compile` (first run will be slow)
- Use GPU if available: `--device cuda`
- Reduce resolution: `--nx 128 --ny 128`
- Disable plotting: `--no-plot`

### NaN Errors

**Problem**: Simulation stops with NaN (Not a Number) errors.

**Solutions**:
- Reduce time step: `--dt 2000` (lower than default)
- Reduce grid resolution
- Check initial conditions and parameters

### Import Errors

**Problem**: Cannot import modules.

**Solutions**:
```bash
# Reinstall the package
uv pip install -e .
```

### Compilation Warnings

**Problem**: Warnings about torch.compile.

**Solution**: This is normal. The `--compile` flag enables experimental features. Either:
- Ignore the warnings (they don't affect results)
- Remove `--compile` flag for cleaner output

---

## Changelog

### Recent Updates (2024)

#### Added Features

**Command-Line Interface**:
- ✅ All example scripts now support argparse for configuration
- ✅ Added `--help` flag to display all options
- ✅ Common options: `--device`, `--compile`, `--nx`, `--ny`, `--output-dir`

**Visualization Options**:
- ✅ `--no-plot`: Disable interactive matplotlib display (batch mode)
- ✅ `--save-images`: Save plots as PNG files to `{output_dir}/images/`
- ✅ `--save-video`: Create MP4 videos using matplotlib's FFMpegWriter
- ✅ `--fps N`: Configure video frame rate (default: 10)
- ✅ `--freq-plot`: Configure plot frequency

**Example-Specific Parameters**:
- **vortexshear.py**: `--Ro`, `--Bu`, `--n-tau`, `--flip-sign`
- **doublegyre.py**: `--model`, `--n-years`, `--wind-mag`, `--slip-coef`, `--freq-save`, `--start-file`, `--layer`
- **natl.py**: All doublegyre options plus `--lat-min/max`, `--data-dir`, `--topo-url`, `--wind-url`

#### Code Improvements

- Fixed LaTeX string escape sequences (using raw strings)
- Reorganized plotting code for better maintainability
- Made compilation optional (disabled by default)
- Improved device selection with 'auto' option
- Standardized output directory structure
- Better progress reporting
- Proper cleanup of matplotlib resources

#### Dependencies

- Added `setuptools>=61.0` to core dependencies (required for torch.compile)
- System dependency: ffmpeg (optional, for video creation)

### Migration from Old Version

**Old approach** (no longer works):
```python
# Edit the script file
nx = 256
ny = 256
```

**New approach**:
```bash
# Pass as command-line arguments
uv run python doublegyre.py --nx 256 --ny 256
```

---

## Citation

If you use this code in your research, please cite the relevant papers referenced in the repository.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or pull request.

---

**Quick Start Summary:**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install package
uv venv
uv pip install -e .

# Run examples
cd examples
uv run python vortexshear.py --help
uv run python doublegyre.py --no-plot --save-video
uv run python natl.py --nx 512 --ny 256 --n-years 1
```

For more detailed examples and usage patterns, all scripts support `--help` to see available options.
