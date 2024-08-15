import contextlib
import importlib

MODULES = {}

with contextlib.suppress(ImportError, NotImplementedError):
    bleh = importlib.import_module("custom_nodes.ComfyUI-bleh")
    bleh_version = getattr(bleh, "BLEH_VERSION", -1)
    if bleh_version < 1:
        raise NotImplementedError
    MODULES["bleh"] = bleh.py

with contextlib.suppress(ImportError, NotImplementedError):
    MODULES["sonar"] = importlib.import_module("custom_nodes.ComfyUI-sonar").py

with contextlib.suppress(ImportError, NotImplementedError):
    MODULES["nnlatentupscale"] = importlib.import_module(
        "custom_nodes.ComfyUi_NNLatentUpscale"
    )

__all__ = ("MODULES",)
