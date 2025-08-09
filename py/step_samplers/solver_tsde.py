import contextlib

import torch
import tqdm

from . import registry
from .solver_base import DESolverStep

HAVE_TSDE = False
with contextlib.suppress(ImportError):
    import torchsde

    HAVE_TSDE = True


class TSDEStep(DESolverStep):
    name = "tsde"
    model_calls = -1
    allow_alt_cfgpp = True
    de_default_solver = "reversible_heun"
    default_eta = 0.0

    def __init__(
        self,
        *args,
        de_initial_step=0.25,
        de_split=1,
        de_adaptive=False,
        tsde_noise_type="scalar",
        tsde_sde_type="stratonovich",
        tsde_levy_area_approx="none",
        tsde_noise_channels=1,
        tsde_g_multiplier=0.05,
        tsde_g_reverse_time=True,
        tsde_g_derp_mode=False,
        tsde_batch_channels=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.de_initial_step = de_initial_step
        self.de_adaptive = de_adaptive
        self.de_split = de_split
        self.de_noise_type = tsde_noise_type
        self.de_sde_type = tsde_sde_type
        self.de_levy_area_approx = tsde_levy_area_approx
        self.de_g_multiplier = tsde_g_multiplier
        self.de_noise_channels = tsde_noise_channels
        self.de_g_reverse_time = tsde_g_reverse_time
        self.de_g_derp_mode = tsde_g_derp_mode
        self.de_batch_channels = tsde_batch_channels

    def check_solver_support(self):
        if not HAVE_TSDE:
            raise RuntimeError(
                "TSDE sampler requires torchsde installed in venv. Example: pip install torchsde"
            )

    def step(self, x):
        s, sn, sigma_down, sigma_up = self.de_get_step(x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x))
        ss = self.ss
        delta = (ss.sigma - sigma_down).item()
        bidx = 0
        mcc = 0
        pbar = None
        _b, c, h, w = x.shape
        outer_self = self

        class SDE(torch.nn.Module):
            noise_type = outer_self.de_noise_type
            sde_type = outer_self.de_sde_type

            @torch.no_grad()
            def f(self, t_rev, y_flat):
                nonlocal mcc
                t = s - (t_rev - sigma_down)
                # print(f"\nf at t_rev={t_rev}, t={t} :: {y_flat.shape}")
                if torch.all(t <= 1e-05).item():
                    return torch.zeros_like(y_flat)
                if mcc >= outer_self.de_max_nfe:
                    raise RuntimeError("TSDEStep: Model call limit exceeded")

                pct = (s - t) / delta
                pbar.n = round(pct.min().item() * 999)
                pbar.update(0)
                pbar.set_description(
                    f"{outer_self.de_solver_name}({mcc}/{outer_self.de_max_nfe})",
                    refresh=True,
                )
                flat_shape = y_flat.shape
                y = y_flat.view(1, c, h, w)
                t32 = t.to(torch.float32)
                del y_flat

                if mcc == 0 and torch.all(t == s):
                    mr_cached = True
                    mr = ss.hcur
                    mcc = 1
                else:
                    mr_cached = False
                    mr = outer_self.call_model(
                        y, t32.clamp(min=1e-05), call_index=mcc, s_in=t.new_ones(1)
                    )
                    mcc += 1
                return -outer_self.to_d(mr)[bidx if mr_cached else 0].view(*flat_shape)

            @torch.no_grad()
            def g(self, t_rev, y_flat):
                t = (s - sigma_down) - (t_rev - sigma_down)
                pct = t / (s - sigma_down)
                if outer_self.de_g_reverse_time:
                    pct = 1.0 - pct
                multiplier = outer_self.de_g_multiplier
                if outer_self.de_g_derp_mode and mcc % 2 == 0:
                    multiplier *= -1
                val = t * pct * multiplier
                if self.noise_type == "diagonal":
                    out = val.repeat(*y_flat.shape)
                elif self.noise_type == "scalar":
                    out = val.repeat(*y_flat.shape, 1)
                else:
                    out = val.repeat(*y_flat.shape, outer_self.de_noise_channels)
                return out

        t = torch.stack((sigma_down, s)).to(torch.float)

        pbar = tqdm.tqdm(
            total=1000, desc=self.de_solver_name, leave=True, disable=ss.disable_status
        )

        dt0 = (
            delta * self.de_initial_step if self.de_adaptive else delta / self.de_split
        )
        results = []
        for batch in tqdm.trange(
            1,
            x.shape[0] + 1,
            desc="batch",
            leave=False,
            disable=x.shape[0] == 1 or ss.disable_status,
        ):
            bidx = batch - 1
            mcc = 0
            sde = SDE()
            if self.de_batch_channels:
                y_flat = x[bidx].flatten(start_dim=1)
            else:
                y_flat = x[bidx].unsqueeze(0).flatten(start_dim=1)
            if sde.noise_type == "diagonal":
                bm_size = (y_flat.shape[0], y_flat.shape[1])
            elif sde.noise_type == "scalar":
                bm_size = (y_flat.shape[0], 1)
            else:
                bm_size = (y_flat.shape[0], self.de_noise_channels)
            bm = torchsde.BrownianInterval(
                dtype=x.dtype,
                device=x.device,
                t0=-s,
                t1=s,
                entropy=ss.noise.seed,
                levy_area_approximation=self.de_levy_area_approx,
                tol=1e-06,
                size=bm_size,
            )

            ys = torchsde.sdeint(
                sde,
                y_flat,
                t,
                method=self.de_solver_name,
                adaptive=self.de_adaptive,
                atol=self.de_atol,
                rtol=self.de_rtol,
                dt=dt0,
                bm=bm,
            )
            del y_flat
            results.append(ys[-1].view(1, c, h, w))
            del ys
        result = torch.cat(results)
        del results

        sigma_up, result = yield from self.adjusted_step(sn, result, mcc, sigma_up)
        if pbar is not None:
            pbar.n = pbar.total
            pbar.update(0)
            pbar.close()
        yield from self.result(result, sigma_up, sigma_down=sigma_down)


registry.add(TSDEStep)
