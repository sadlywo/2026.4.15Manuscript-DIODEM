from __future__ import annotations

"""Best-effort Torch import helpers.

This module allows the rest of the project to be imported even when the local
Torch runtime is temporarily unhealthy. Training code still fails loudly when
Torch is actually required.
"""

TORCH_IMPORT_ERROR = None

try:  # pragma: no cover - depends on local runtime
    import torch
    from torch import nn
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised on broken runtimes
    torch = None
    nn = None
    Dataset = object
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = exc


def require_torch() -> None:
    """Raise a descriptive error if Torch cannot be imported."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for this operation, but it could not be imported. "
            f"Original error: {TORCH_IMPORT_ERROR}"
        )

