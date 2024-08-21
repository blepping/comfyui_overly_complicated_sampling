from .py import nodes
from .py import custom_noise

NODE_CLASS_MAPPINGS = {
    "OCS Sampler": nodes.SamplerNode,
    "OCS Substeps": nodes.SubstepsNode,
    "OCS Group": nodes.GroupNode,
    "OCS Param": nodes.ParamNode,
    "OCS MultiParam": nodes.MultiParamNode,
    "OCS ModelSetMaxSigma": nodes.ModelSetMaxSigmaNode,
    "OCS SimpleRestartSchedule": nodes.SimpleRestartSchedule,
} | custom_noise.NODE_CLASS_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS"]
