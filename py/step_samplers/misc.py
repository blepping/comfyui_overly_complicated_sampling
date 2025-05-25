from .base import SingleStepSampler, registry


# Referenced from https://github.com/ace-step/ACE-Step/
class PingPongStep(SingleStepSampler):
    name = "pingpong"
    default_eta = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pingpong_options = self.options.pop("pingpong", {})
        self.pingpong_start_step = pingpong_options.get("start_step", 0)
        self.pingpong_end_step = pingpong_options.get("end_step", 0)

    def step(self, x):
        ss = self.ss
        use_pingpong = self.pingpong_start_step <= ss.step <= self.pingpong_end_step
        if not use_pingpong:
            return (yield from self.euler_step(x, eta=0.0))
        sn = ss.sigma_next
        denoised = (
            ss.denoised * (1.0 - sn) if ss.model.is_rectified_flow else ss.denoised
        )
        yield from self.result(denoised, sn, sigma_down=sn)


registry.add(
    PingPongStep,
)
