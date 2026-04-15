from __future__ import annotations

import os
import random

import numpy as np

from project.utils.torch_compat import TORCH_AVAILABLE, torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch when available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if TORCH_AVAILABLE:  # pragma: no cover - depends on local runtime
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

