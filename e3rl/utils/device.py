"""Device resolution helpers."""

from __future__ import annotations

import torch


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def resolve_device(device: str | None = None) -> str:
    """Return a torch device string, validating availability.

    With ``device=None``, picks the best available backend in the order
    CUDA -> MPS -> CPU. Otherwise, validates that the requested backend is
    available on this machine and returns it unchanged.
    """
    if device is None:
        if _cuda_available():
            return "cuda:0"
        if _mps_available():
            return "mps"
        return "cpu"

    backend = device.split(":", 1)[0]

    if backend == "cuda":
        if not _cuda_available():
            raise RuntimeError(f"Requested device {device!r} but CUDA is not available.")
        return device
    if backend == "mps":
        if not _mps_available():
            raise RuntimeError(f"Requested device {device!r} but MPS is not available.")
        return device
    if backend == "cpu":
        return device

    raise ValueError(f"Unknown device backend in {device!r}; expected cuda, mps, or cpu.")
