"""PyTorch compatibility shims for older versions.

This module provides backports for features added in newer PyTorch versions.
Import this module early to ensure compatibility shims are applied before
any model loading code runs.

Specifically:
- set_submodule: Added in PyTorch 2.5, required by bitsandbytes for quantization.
  We're on PyTorch 2.4.1 because newer versions dropped support for SM 6.1 (Tesla P40).
"""

from __future__ import annotations

import logging

import torch
from torch.nn import Module

logger = logging.getLogger(__name__)

_PATCHED = False


def _set_submodule(self: Module, target: str, module: Module) -> None:
    """Set a submodule given its dotted path.

    This is a backport of torch.nn.Module.set_submodule() from PyTorch 2.5+.

    Args:
        target: Dot-separated path to the submodule (e.g., "layer.fc")
        module: The module to set at the target path
    """
    atoms = target.split(".")
    mod = self
    for atom in atoms[:-1]:
        mod = getattr(mod, atom)
    setattr(mod, atoms[-1], module)


def apply_patches() -> None:
    """Apply compatibility patches to torch.nn.Module if needed."""
    global _PATCHED
    if _PATCHED:
        return

    if not hasattr(Module, "set_submodule"):
        logger.info(
            "Patching torch.nn.Module.set_submodule for PyTorch %s compatibility",
            torch.__version__,
        )
        Module.set_submodule = _set_submodule

    _PATCHED = True


# Apply patches on import
apply_patches()
