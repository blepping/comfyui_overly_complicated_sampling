from .py import nodes


NODE_CLASS_MAPPINGS = {
    "ComposableSampler": nodes.ComposableSampler,
    "ComposableStepSampler": nodes.ComposableStepSampler,
}
__all__ = ["NODE_CLASS_MAPPINGS"]
