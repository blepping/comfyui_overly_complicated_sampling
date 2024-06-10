from .py import nodes


NODE_CLASS_MAPPINGS = {
    "ComposableSampler": nodes.ComposableSampler,
    "ComposableStepSampler": nodes.ComposableStepSampler,
    "SubstepsGroup": nodes.SubstepsGroup,
    "CSamplerParam": nodes.CSamplerParam,
    "CSamplerParamMulti": nodes.CSamplerParamMulti,
}
__all__ = ["NODE_CLASS_MAPPINGS"]
