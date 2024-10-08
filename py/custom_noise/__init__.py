from . import noise_perlin
from . import nodes

NODE_CLASS_MAPPINGS = {
    "OCSNoise PerlinSimple": noise_perlin.PerlinSimpleNode,
    "OCSNoise PerlinAdvanced": noise_perlin.PerlinAdvancedNode,
    "OCSNoise to SONAR_CUSTOM_NOISE": nodes.ToSonarNode,
}
