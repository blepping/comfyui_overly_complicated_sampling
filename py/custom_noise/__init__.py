from . import nodes

NODE_CLASS_MAPPINGS = {
    "OCSNoise PerlinSimple": nodes.PerlinSimpleNode,
    "OCSNoise PerlinAdvanced": nodes.PerlinAdvancedNode,
    "OCSNoise to SONAR_CUSTOM_NOISE": nodes.ToSonarNode,
    "OCSNoise Conditioning": nodes.NoiseConditioningNode,
    "OCSNoise OverrideSamplerNoise": nodes.SamplerNodeConfigOverride,
}
