from __future__ import annotations

import os
from pathlib import Path

import torch

_LOADED = False
_LIB_BASENAMES = [
    "rt2d_torch.dll",
    "librt2d_torch.so",
    "rt2d_torch.so",
    "librt2d_torch.dylib",
]


def _iter_candidate_paths() -> list[Path]:
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]

    candidates: list[Path] = []
    for lib_name in _LIB_BASENAMES:
        candidates.append(package_dir / lib_name)

    build_dir = project_root / "build"
    if build_dir.exists():
        for path in build_dir.rglob("*rt2d_torch*"):
            if path.is_file() and path.suffix.lower() in {".dll", ".so", ".dylib"}:
                candidates.append(path)

    return candidates


def load_library(path: str | os.PathLike[str] | None = None) -> None:
    global _LOADED
    if _LOADED:
        return

    if path is not None:
        torch.ops.load_library(str(path))
        _LOADED = True
        return

    env_path = os.environ.get("RT2D_TORCH_LIB")
    if env_path:
        torch.ops.load_library(env_path)
        _LOADED = True
        return

    for candidate in _iter_candidate_paths():
        if candidate.exists():
            torch.ops.load_library(str(candidate))
            _LOADED = True
            return

    raise FileNotFoundError(
        "Could not find rt2d_torch shared library. Build with CMake and install to python/rt2d "
        "or set RT2D_TORCH_LIB to the compiled library path."
    )

