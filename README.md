# 2D-RT: C/C++/CUDA/Python Ray Tracing Project (CMake)

This repository is a starter project for propagation-domain ray tracing with:

- C/C++/CUDA mixed implementation
- GPU-first execution to minimize host-device transfers
- Torch custom operator integration (`torch.ops.load_library`)
- CMake as the only build system
- Optional GitHub open-source dependencies via `FetchContent`

## Project Layout

```text
.
|-- CMakeLists.txt
|-- include/rt2d/
|   |-- c_scene.h
|   `-- raytrace.h
|-- src/
|   |-- c/c_scene.c
|   |-- cpp/ops.cpp
|   |-- cpp/raytrace_cpu.cpp
|   `-- cuda/raytrace_cuda.cu
|-- python/
|   |-- rt2d/__init__.py
|   |-- rt2d/loader.py
|   `-- examples/demo.py
`-- scripts/build.ps1
```

## Build Requirements

- Python 3.10+
- PyTorch installed in the active Python environment
- CMake 3.24+
- C++ compiler with C++17 support
- CUDA toolkit + NVIDIA driver (for GPU path)

## Build

From project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Config Release
```

If you need a specific Python environment (recommended for PyTorch), pass:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Config Release -PythonExe "C:\path\to\python.exe"
```

This will:

1. Configure CMake
2. Build `rt2d_torch`
3. Install the shared library to `python/rt2d`

## Python Usage

```python
import torch
import rt2d

device = "cuda" if torch.cuda.is_available() else "cpu"
origins = torch.zeros((4096, 3), device=device, dtype=torch.float32)
dirs = torch.randn((4096, 3), device=device, dtype=torch.float32)
dirs = dirs / dirs.norm(dim=1, keepdim=True)
center = torch.tensor([0.0, 0.0, 3.0], device=device, dtype=torch.float32)

dist = rt2d.raytrace(origins, dirs, center, sphere_radius=1.0)
```

Run sample:

```powershell
$env:PYTHONPATH = "$PWD\python"
python .\python\examples\demo.py
```

## GPU-First Dataflow

- Inputs are expected to already be CUDA tensors when using GPU.
- The CUDA kernel computes intersections fully on-device.
- Output is created directly on GPU.
- No mandatory host round-trip in CUDA path.

## Optional GitHub Dependencies

You can enable open-source dependencies fetched from GitHub:

- `fmt` for formatting/logging
- `glm` for math primitives

Enable by configuring with:

```powershell
cmake -S . -B build -DRT2D_USE_FETCH_DEPS=ON
```

## Notes

- Operator name is `rt2d::raytrace`, exposed to Python as `torch.ops.rt2d.raytrace`.
- CPU and CUDA dispatch are both registered.
- Current example traces rays against a sphere to provide a minimal and testable foundation.
