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
    "OCS ApplyExpressionLatent": nodes.ApplyExpressionLatent,
    "OCS ApplyExpressionImage": nodes.ApplyExpressionImage,
} | custom_noise.NODE_CLASS_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS"]
