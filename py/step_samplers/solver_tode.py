import contextlib

import torch
import tqdm

from . import registry
from .solver_base import DESolverStep


HAVE_TODE = False
with contextlib.suppress(ImportError, RuntimeError):
    import torchode as tode

    HAVE_TODE = True


class TODEStep(DESolverStep):
    name = "tode"
    model_calls = -1
    allow_alt_cfgpp = True
    de_default_solver = "dopri5"
    default_eta = 0.0

    def __init__(
        self,
        *args,
        de_initial_step=0.25,
        tode_compile=False,
        de_ctl_pcoeff=0.3,
        de_ctl_icoeff=0.9,
        de_ctl_dcoeff=0.2,
        **kwargs,
    ):
        if not HAVE_TODE:
            raise RuntimeError(
                "TODE sampler requires torchode installed in venv. Example: pip install torchode"
            )
        super().__init__(*args, **kwargs)
        self.de_solver_method = tode.interface.METHODS[self.de_solver_name]
        self.de_ctl_pcoeff = de_ctl_pcoeff
        self.de_ctl_icoeff = de_ctl_icoeff
        self.de_ctl_dcoeff = de_ctl_dcoeff
        self.de_compile = tode_compile
        self.de_initial_step = de_initial_step

    def check_solver_support(self):
        if not HAVE_TODE:
            raise RuntimeError(
                "TODE sampler requires torchode installed in venv. Example: pip install torchode"
            )

    def step(self, x):
        s, sn, sigma_down, sigma_up = self.de_get_step(x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x))
        ss = self.ss
        delta = (ss.sigma - sigma_down).item()
        mcc = 0
        pbar = None
        b, c, h, w = x.shape

        def odefn(t, y_flat):
            nonlocal mcc
            if torch.all(t <= 1e-05).item():
                return torch.zeros_like(y_flat)
            if mcc >= self.de_max_nfe:
                raise RuntimeError("TDEStep: Model call limit exceeded")

            pct = (s - t) / delta
            pbar.n = round(pct.min().item() * 999)
            pbar.update(0)
            pbar.set_description(
                f"{self.de_solver_name}({mcc}/{self.de_max_nfe})", refresh=True
            )
            y = y_flat.reshape(-1, c, h, w)
            t32 = t.to(torch.float32)
            del y_flat

            if mcc == 0 and torch.all(t == s):
                mr = ss.hcur
                mcc = 1
            else:
                mr = self.call_model(y, t32.clamp(min=1e-05), call_index=mcc)
                mcc += 1
            result = self.to_d(mr).flatten(start_dim=1)
            for bi in range(t.shape[0]):
                if t[bi] <= 1e-05:
                    result[bi, :] = 0
            return result

        t = torch.stack((s, sigma_down)).to(torch.float64).repeat(b, 1)

        pbar = tqdm.tqdm(
            total=1000, desc=self.de_solver_name, leave=True, disable=ss.disable_status
        )

        term = tode.ODETerm(odefn)
        method = self.de_solver_method(term=term)
        controller = tode.PIDController(
            term=term,
            atol=self.de_atol,
            rtol=self.de_rtol,
            dt_min=1e-05,
            pcoeff=self.de_ctl_pcoeff,
            icoeff=self.de_ctl_icoeff,
            dcoeff=self.de_ctl_dcoeff,
        )
        solver_ = tode.AutoDiffAdjoint(method, controller)
        solver = solver_ if not self.de_compile else torch.compile(solver_)
        problem = tode.InitialValueProblem(
            y0=x.flatten(start_dim=1), t_start=t[:, 0], t_end=t[:, -1]
        )
        dt0 = (
            (t[:, -1] - t[:, 0]) * self.de_initial_step
            if self.de_initial_step
            else None
        )
        solution = solver.solve(problem, dt0=dt0)

        # print("\nSOLUTION", solution.stats, solution.ys.shape)
        result = solution.ys[:, -1].reshape(-1, c, h, w)
        del solution

        sigma_up, result = yield from self.adjusted_step(sn, result, mcc, sigma_up)
        if pbar is not None:
            pbar.n = pbar.total
            pbar.update(0)
            pbar.close()
        yield from self.result(result, sigma_up, sigma_down=sigma_down)


registry.add(TODEStep)
