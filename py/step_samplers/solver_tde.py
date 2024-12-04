import contextlib

import torch
import tqdm

from .solver_base import DESolverStep

HAVE_TDE = False
with contextlib.suppress(ImportError):
    import torchdiffeq as tde

    HAVE_TDE = True


class TDEStep(DESolverStep):
    name = "tde"
    model_calls = -1
    allow_alt_cfgpp = True
    allow_cfgpp = False
    de_default_solver = "rk4"
    default_eta = 0.0

    def __init__(
        self,
        *args,
        de_split=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.de_split = de_split

    def check_solver_support(self):
        if not HAVE_TDE:
            raise RuntimeError(
                "TDE sampler requires torchdiffeq installed in venv. Example: pip install torchdiffeq"
            )

    def step(self, x):
        s, sn, sigma_down, sigma_up = self.de_get_step(x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x))
        ss = self.ss
        delta = (s - sigma_down).item()
        mcc = 0
        bidx = 0
        pbar = None

        def odefn(t, y):
            nonlocal mcc
            if t < 1e-05:
                return torch.zeros_like(y)
            if mcc >= self.de_max_nfe:
                raise RuntimeError("TDEStep: Model call limit exceeded")

            pct = (s - t) / delta
            pbar.n = round(min(999, pct.item() * 999))
            pbar.update(0)
            pbar.set_description(
                f"{self.de_solver_name}({mcc}/{self.de_max_nfe})", refresh=True
            )

            if t == ss.sigma and torch.equal(x[bidx], y):
                mr_cached = True
                mr = ss.hcur
                mcc = 1
            else:
                mr_cached = False
                mr = self.call_model(
                    y.unsqueeze(0), t, call_index=mcc, s_in=t.new_ones(1)
                )
                mcc += 1
            return self.to_d(mr)[bidx if mr_cached else 0]

        result = torch.zeros_like(x)
        t = sigma_down.new_zeros(self.de_split + 1)
        torch.linspace(ss.sigma, sigma_down, t.shape[0], out=t)

        for batch in tqdm.trange(
            1,
            x.shape[0] + 1,
            desc="batch",
            leave=False,
            disable=x.shape[0] == 1 or ss.disable_status,
        ):
            bidx = batch - 1
            mcc = 0
            if pbar is not None:
                pbar.close()
            pbar = tqdm.tqdm(
                total=1000,
                desc=self.de_solver_name,
                leave=True,
                disable=ss.disable_status,
            )
            solution = tde.odeint(
                odefn,
                x[bidx],
                t,
                rtol=self.de_rtol,
                atol=self.de_atol,
                method=self.de_solver_name,
                options={
                    "min_step": 1e-05,
                    "dtype": torch.float64,
                },
            )[-1]
            result[bidx] = solution

        sigma_up, result = yield from self.adjusted_step(sn, result, mcc, sigma_up)
        if pbar is not None:
            pbar.n = pbar.total
            pbar.update(0)
            pbar.close()
        yield from self.result(result, sigma_up, sigma_down=sigma_down)
