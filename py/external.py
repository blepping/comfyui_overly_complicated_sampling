import contextlib
import importlib
import sys


def get_custom_node(name):
    module_key = f"custom_nodes.{name}"
    try:
        spec = importlib.util.find_spec(module_key)
        if spec is None:
            raise ModuleNotFoundError(module_key)
        module = next(
            v
            for v in sys.modules.copy().values()
            if hasattr(v, "__spec__")
            and v.__spec__ is not None
            and v.__spec__.origin == spec.origin
        )
    except StopIteration:
        raise ModuleNotFoundError(module_key) from None
    return module


class Integrations:
    def __init__(self):
        self.initialized = False
        self.modules = {}
        self.init_handlers = []

    def __getitem__(self, key):
        return self.modules[key]

    def __contains__(self, key):
        return key in self.modules

    def __getattr__(self, key):
        return self.modules.get(key)

    def register_init_handler(self, fun):
        self.init_handlers.append(fun)

    def initialize(self) -> None:
        if self.initialized:
            return
        self.initialized = True
        with contextlib.suppress(ModuleNotFoundError):
            self.modules["tiled_diffusion"] = get_custom_node("ComfyUI-TiledDiffusion")

        with contextlib.suppress(ModuleNotFoundError):
            self.modules["sonar"] = get_custom_node("ComfyUI-sonar").py

        with contextlib.suppress(ModuleNotFoundError):
            self.modules["nnlatentupscale"] = get_custom_node("ComfyUi_NNLatentUpscale")

        with contextlib.suppress(ModuleNotFoundError, NotImplementedError):
            bleh = get_custom_node("ComfyUI-bleh")
            bleh_version = getattr(bleh, "BLEH_VERSION", -1)
            if bleh_version < 1:
                raise NotImplementedError
            self.modules["bleh"] = bleh.py

        for init_handler in self.init_handlers:
            init_handler()


MODULES = Integrations()


__all__ = ("MODULES",)
