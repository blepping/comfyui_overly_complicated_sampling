class ToSonarNode:
    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    CATEGORY = "OveryComplicatedSampling/noise"
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ocs_noise": ("OCS_NOISE",),
            },
        }

    @classmethod
    def go(cls, ocs_noise):
        return (ocs_noise,)
