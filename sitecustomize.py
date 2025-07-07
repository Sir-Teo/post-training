"""Site customisation module automatically imported by Python.

We stub out `torchvision` so that transformer imports that *optionally* reference
vision utilities don't crash on systems where torchvision is either absent or
compiled without the necessary CPU/CUDA ops (common on macOS CPU installs).

The demo scripts in this repo are text-only and do not rely on any vision
features, so a minimal stub is perfectly sufficient.
"""
import sys
import types

if "torchvision" not in sys.modules:
    stub = types.ModuleType("torchvision")
    # create common sub-modules transformers tries to access
    stub.transforms = types.ModuleType("torchvision.transforms")
    stub.datasets = types.ModuleType("torchvision.datasets")
    sys.modules["torchvision"] = stub
    sys.modules["torchvision.transforms"] = stub.transforms
    sys.modules["torchvision.datasets"] = stub.datasets
