import torch


class Normalize:
    def __init__(
        self,
        start_step=0,
        end_step=9999,
        phase="after",
        adjust_target=0,
        balance_scale=1.0,
        adjust_scale=1.0,
        dims=(-2, -1),
    ):
        self.start_step = start_step
        self.end_step = end_step
        self.phase = phase.lower().strip()  # before, after, all
        if isinstance(adjust_target, str):
            adjust_target = adjust_target.lower().strip()
            if adjust_target not in ("x",):
                raise ValueError("Bad target mean")
        # "x", scalar or array matching mean dims
        self.adjust_target = adjust_target
        # multiplier on adjustment, scalar or array matching mean dims
        self.adjust_scale = adjust_scale
        self.balance_scale = balance_scale
        self.dims = dims

    def __call__(self, ss, sigma, latent, phase, orig_x=None):
        if ss.step < self.start_step or ss.step > self.end_step:
            return latent
        if self.phase != "all" and phase != self.phase:
            return latent
        if self.adjust_target == "x" and orig_x is None:
            raise ValueError("Can only use source x in after phase")
        adjust_scale, balance_scale = (
            torch.tensor(v, dtype=latent.dtype).to(latent)
            if isinstance(v, (list, tuple))
            else v
            for v in (self.adjust_scale, self.balance_scale)
        )
        latent_mean = latent.mean(dim=self.dims, keepdim=True)
        # print("MEAN", latent_mean)
        latent = latent - latent_mean * balance_scale
        if self.adjust_target == "x":
            latent += orig_x.mean(dim=self.dims, keepdim=True) * adjust_scale
        elif isinstance(self.adjust_target, (list, tuple)):
            adjust_target = torch.tensor(self.adjust_target, dtype=latent.dtype).to(
                latent
            )
            latent += adjust_target * adjust_scale
        else:
            latent += self.adjust_target * adjust_scale
        return latent
