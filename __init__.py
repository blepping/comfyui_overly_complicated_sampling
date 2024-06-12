from .py import nodes


NODE_CLASS_MAPPINGS = {
    "OCS Sampler": nodes.SamplerNode,
    "OCS Substeps": nodes.SubstepsNode,
    "OCS Group": nodes.GroupNode,
    "OCS Param": nodes.ParamNode,
    "OCS MultiParam": nodes.MultiParamNode,
}
__all__ = ["NODE_CLASS_MAPPINGS"]
