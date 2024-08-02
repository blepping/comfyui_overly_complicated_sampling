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
    sonar = importlib.import_module("custom_nodes.ComfyUI-sonar")
    MODULES["sonar"] = sonar.py


__all__ = ("MODULES",)
