"""Shared helpers for wrappers that swap a base model's energy head.

`AlphaInjectionWrapper`, `ScalarEnsembleWrapper`, and `ScalarDropoutWrapper`
all need to (i) discover the input dim of the base model's energy/curve head
and (ii) replace that head with a stub that captures the graph representation
`z` for the wrapper's own head. This module factors that logic out.
"""

from typing import Iterable, Optional

import torch
import torch.nn as nn


HEAD_ATTR_NAMES: tuple = ("energy_head", "head", "curve_head", "energy")


def head_attr_candidates(names: Iterable[str] = HEAD_ATTR_NAMES) -> list:
    """Dotted attribute paths to try when locating the energy head."""
    out = []
    for name in names:
        out.append(name)
        out.append(f"backbone.{name}")
    return out


def detect_head_input_dim(base: nn.Module,
                          names: Iterable[str] = HEAD_ATTR_NAMES,
                          default: Optional[int] = None) -> int:
    """Return the in_features of the base model's energy head, if found.

    Walks the candidate attribute paths in `names` (plus `backbone.<name>`).
    Falls back to `module.named_modules()` matching the leaf component
    against `names` exactly (so `rms_head`/`cbr_head` are not picked).
    Returns `default` if nothing matches; raises if `default is None`.
    """
    name_set = tuple(names)
    for attr_path in head_attr_candidates(name_set):
        obj = base
        ok = True
        for part in attr_path.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if not ok:
            continue
        in_features = _head_in_features(obj)
        if in_features is not None:
            return in_features
    for name, mod in base.named_modules():
        leaf = name.split('.')[-1]
        if leaf in name_set:
            in_features = _head_in_features(mod)
            if in_features is not None:
                return in_features
    if default is None:
        raise RuntimeError(
            f"Could not locate an energy head on {type(base).__name__}; "
            f"tried attribute names {name_set}.")
    return default


def _head_in_features(obj: nn.Module) -> Optional[int]:
    if isinstance(obj, nn.Sequential):
        for m in obj:
            if isinstance(m, nn.Linear):
                return m.in_features
    elif isinstance(obj, nn.Linear):
        return obj.in_features
    return None


class CaptureZ(nn.Module):
    """Stub that records its input and returns a zero curve of length K.

    Installed in place of a base model's energy head so the wrapper can grab
    the graph representation z, then run the wrapper's own α-conditioned
    head over [z || α].
    """

    def __init__(self, K: int):
        super().__init__()
        self.K = K
        self.z: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.z = x
        return torch.zeros(x.size(0), self.K, device=x.device)


def install_capture_head(base: nn.Module, capture: CaptureZ,
                         names: Iterable[str] = HEAD_ATTR_NAMES) -> str:
    """Replace `base`'s energy head with `capture`. Returns the attr path used."""
    name_set = tuple(names)
    # Direct attributes first (faster, common case).
    for name in name_set:
        if hasattr(base, name):
            setattr(base, name, capture)
            return name
        bb = getattr(base, 'backbone', None)
        if bb is not None and hasattr(bb, name):
            setattr(bb, name, capture)
            return f"backbone.{name}"
    # Walk submodules as a last resort.
    for path, _mod in base.named_modules():
        leaf = path.split('.')[-1]
        if leaf in name_set:
            owner = base
            for part in path.split('.')[:-1]:
                owner = getattr(owner, part)
            setattr(owner, leaf, capture)
            return path
    raise RuntimeError(
        f"Could not install capture head on {type(base).__name__}; "
        f"tried attribute names {name_set}.")
