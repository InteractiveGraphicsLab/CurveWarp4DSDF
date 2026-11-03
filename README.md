# Curve-based Interactive Refinement for Spatiotemporal Segmentation by Deforming 4D Neural SDF



## 1. Environment Setup

### Prerequisites
- Python 3.8
- CUDA 11.8 (Ensure `nvcc` and CUDA Toolkit are installed for C++ extension compilation)
- C++ Compiler (Visual Studio 2019/2022 with *"Desktop development with C++"* for Windows)

### Installation
Create and activate a virtual environment using standard `venv`:
```bash
# Clone or extract the repository, then navigate to the project root
cd /path/to/project

# Create a Python 3.8 virtual environment
# On Linux:
python3.8 -m venv venv
# On Windows (PowerShell / CMD):
py -3.8 -m venv venv

# Activate the virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\activate

# Install PyTorch with CUDA 11.8 support
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install tiny-cuda-nn (Requires CUDA Toolkit & C++ compiler)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install required dependencies
pip install SimpleITK==2.3.1 scipy pyyaml
```



## 2. Tested Environment

We tested our implementation under the following hardware and software configuration:

- **Operating System:** Windows 11
- **CPU:** AMD Ryzen 5 7600X 6-Core Processor
- **System Memory (RAM):** 32 GB
- **GPU:** NVIDIA GeForce RTX 3060 (12 GB)
- **CUDA Toolkit:** 11.8
- **Python:** 3.8.10
- **PyTorch:** 2.1.2+cu118
- **Torchvision:** 0.16.2+cu118



## 3. How to Run

To execute data preparation, 4D SDF pre-training, and deformation optimization:
```bash
python main.py --config .\data\config_example.yaml
```

### Key Configuration Parameters (`config_example.yaml`)
- `input_dir / output_dir` : Paths for input masks and output refined volumes.
- `target_id`: Integer label ID corresponding to the target segmentation region.
- `num_output_frames`: Temporal Resampling / Interpolation Control.
  - If set to null, the output frame count matches the input length.
  - Setting this to a custom integer enables continuous temporal interpolation, leveraging the continuous spatiotemporal 4D SDF representation to output volumes at higher temporal resolution.
- `curves`: Control point curve constraints specified in physical coordinates [phys_x, phys_y, phys_z, frame_index].

### Expected Output Behavior
- **Input:** 10 sequential 3D binary segmentation mask files in MetaImage (`.mha`) format representing a spherical 4D volume, alongside curve constraints applied exclusively to specific target frames (Frame 0 and Frame 5).
- **Output:** Only the selected target frames undergoes deformation corresponding to the curve constraints, while unconstrained frames preserve their initial state.

### Generating a Configuration File
To generate a configuration file for your own data, use the **Export config** button in **RoiPainter4D → ModeRefCurveDeform**, available in the [RoiPainter repository](https://github.com/InteractiveGraphicsLab/RoiPainter).

The exported configuration file can be used to run this project:

```bash
python main.py --config /path/to/exported_config.yaml
```



## 4. Visualizing Output Files

Our implementation generates 3D volumetric binary segmentation masks in MetaImage (`.mha`) format, which preserve physical spacing and orientation metadata.

The generated segmentation masks can be visualized in 3D using **RoiPainter4D**, available in the [RoiPainter repository](https://github.com/InteractiveGraphicsLab/RoiPainter).

Alternatively, the output files can be visualized using
To visualize segmentation masks in 3D using **3D Slicer**:
1. Download and install [3D Slicer](https://download.slicer.org/)
2. Drag and drop the `.mha` file into 3D Slicer.
3. Check Show Options in the Add Data window, check the LabelMap option for the .mha file, and click OK
4. Select Volume Rendering from the top module dropdown menu.
5. Click the eye icon (Visibility) at the top of the module panel to display the target mask in 3D.
