from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import rt2d


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    n = 4096
    origins = torch.zeros((n, 3), device=device, dtype=torch.float32)
    directions = torch.randn((n, 3), device=device, dtype=torch.float32)
    directions = directions / directions.norm(dim=1, keepdim=True)
    sphere_center = torch.tensor([0.0, 0.0, 3.0], device=device, dtype=torch.float32)

    distances = rt2d.raytrace(origins, directions, sphere_center, sphere_radius=1.0)
    valid = distances > 0
    print(f"hits: {int(valid.sum().item())}/{n}")
    if valid.any():
        print(f"mean hit distance: {float(distances[valid].mean().item()):.4f}")


if __name__ == "__main__":
    main()
