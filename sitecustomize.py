"""Site customisation module automatically imported by Python.

We stub out `torchvision` so that transformer imports that *optionally* reference
vision utilities don't crash on systems where torchvision is either absent or
compiled without the necessary CPU/CUDA ops (common on macOS CPU installs).

The demo scripts in this repo are text-only and do not rely on any vision
features, so a minimal stub is perfectly sufficient.
"""
import sys
import types

# Always stub torchvision to avoid heavy/broken vision deps.
stub = types.ModuleType("torchvision")
stub.transforms = types.ModuleType("torchvision.transforms")
# Minimal placeholder for torchvision.transforms.InterpolationMode enum used by transformers
class _InterpolationMode:
    BILINEAR = 2
    BICUBIC = 3
    NEAREST = 0

stub.transforms.InterpolationMode = _InterpolationMode
stub.datasets = types.ModuleType("torchvision.datasets")
# Replace/insert into sys.modules unconditionally
for name in ("torchvision", "torchvision.transforms", "torchvision.datasets"):
    sys.modules[name] = getattr(stub, name.split(".")[-1], stub) if "." in name else stub
