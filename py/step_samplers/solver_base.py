from .base import SingleStepSampler, MinSigmaStepMixin


class DESolverStep(SingleStepSampler, MinSigmaStepMixin):
    de_default_solver = None
    sample_sigma_zero = True
    default_eta = 0.0

    def __init__(
        self,
        *args,
        de_solver=None,
        de_max_nfe=100,
        de_rtol=-2.5,
        de_atol=-3.5,
        de_fixup_hack=0.025,
        de_split=1,
        de_min_sigma=0.0292,
        **kwargs,
    ):
        self.check_solver_support()
        super().__init__(*args, **kwargs)
        de_solver = self.de_default_solver if de_solver is None else de_solver
        self.de_solver_name = de_solver
        self.de_max_nfe = de_max_nfe
        self.de_rtol = 10**de_rtol
        self.de_atol = 10**de_atol
        self.de_fixup_hack = de_fixup_hack
        self.de_split = de_split
        self.de_min_sigma = de_min_sigma if de_min_sigma is not None else 0.0

    def check_solver_support(self):
        raise NotImplementedError

    def de_get_step(self, x):
        eta = self.get_dyn_eta()
        ss = self.ss
        s, sn = ss.sigma, ss.sigma_next
        sn = self.adjust_step(sn, self.de_min_sigma)
        sigma_down, sigma_up = self.get_ancestral_step(eta, sigma_next=sn)
        if self.de_fixup_hack != 0:
            sigma_down = (sigma_down - (s - sigma_down) * self.de_fixup_hack).clamp(
                min=0
            )
        return s, sn, sigma_down, sigma_up

    @staticmethod
    def reverse_time(t, t0, t1):
        return t1 + (t0 - t)
