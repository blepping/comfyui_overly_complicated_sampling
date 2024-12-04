import contextlib
import os
import typing
import warnings

import numpy
import torch
import tqdm

import comfy

from .solver_base import DESolverStep

HAVE_DIFFRAX = False


if not os.environ.get("COMFYUI_OCS_NO_DIFFRAX_SOLVER"):
    with contextlib.suppress(ImportError):
        import diffrax
        import jax

        if not os.environ.get("COMFYUI_OCS_NO_DISABLE_JAX_PREALLOCATE"):
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # jax.config.update("jax_enable_x64", True)

        HAVE_DIFFRAX = True


if HAVE_DIFFRAX:

    class RevVirtualBrownianTree(diffrax.VirtualBrownianTree):
        def evaluate(self, t0, t1, *args, **kwargs):
            if t1 is not None:
                return super().evaluate(t1, t0, *args, **kwargs)
            return super().evaluate(t0, t1, *args, **kwargs)

    class StepCallbackTqdmProgressMeter(diffrax.TqdmProgressMeter):
        step_callback: typing.Callable = None

        def _init_bar(self, *args, **kwargs):
            if self.step_callback is None:
                return super()._init_bar(*args, **kwargs)
            bar_format = "{percentage:.2f}%{step_callback}|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            step_callback = self.step_callback

            class WrapTqdm(tqdm.tqdm):
                @property
                def format_dict(self):
                    d = super().format_dict
                    d.update(step_callback=step_callback())
                    return d

            return WrapTqdm(total=100, unit="%", bar_format=bar_format)


class DiffraxStep(DESolverStep):
    name = "diffrax"
    model_calls = -1
    allow_alt_cfgpp = True
    de_default_solver = "dopri5"
    default_eta = 0.0

    def __init__(
        self,
        *args,
        de_split=1,
        de_initial_step=0.25,
        de_ctl_pcoeff=0.3,
        de_ctl_icoeff=0.9,
        de_ctl_dcoeff=0.2,
        diffrax_adaptive=False,
        diffrax_fake_pure_callback=True,
        diffrax_g_multiplier=0.0,
        diffrax_half_solver=False,
        diffrax_batch_channels=False,
        diffrax_levy_area_approx="brownian_increment",
        diffrax_error_order=None,
        diffrax_sde_mode=False,
        diffrax_g_reverse_time=False,
        diffrax_g_time_scaling=False,
        diffrax_g_split_time_mode=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        solvers = dict(
            euler=diffrax.Euler,
            heun=diffrax.Heun,
            midpoint=diffrax.Midpoint,
            ralston=diffrax.Ralston,
            bosh3=diffrax.Bosh3,
            tsit5=diffrax.Tsit5,
            dopri5=diffrax.Dopri5,
            dopri8=diffrax.Dopri8,
            implicit_euler=diffrax.ImplicitEuler,
            # kvaerno3=diffrax.Kvaerno3,
            # kvaerno4=diffrax.Kvaerno4,
            # kvaerno5=diffrax.Kvaerno5,
            semi_implicit_euler=diffrax.SemiImplicitEuler,
            reversible_heun=diffrax.ReversibleHeun,
            leapfrog_midpoint=diffrax.LeapfrogMidpoint,
            euler_heun=diffrax.EulerHeun,
            ito_milstein=diffrax.ItoMilstein,
            stratonovich_milstein=diffrax.StratonovichMilstein,
            sea=diffrax.SEA,
            sra1=diffrax.SRA1,
            shark=diffrax.ShARK,
            general_shark=diffrax.GeneralShARK,
            slow_rk=diffrax.SlowRK,
            spark=diffrax.SPaRK,
        )
        levy_areas = dict(
            brownian_increment=diffrax.BrownianIncrement,
            space_time=diffrax.SpaceTimeLevyArea,
            space_time_time=diffrax.SpaceTimeTimeLevyArea,
        )
        # jax.config.update("jax_disable_jit", True)
        self.de_solver_method = solvers[self.de_solver_name]()
        if diffrax_half_solver:
            self.de_solver_method = diffrax.HalfSolver(self.de_solver_method)
        self.de_ctl_pcoeff = de_ctl_pcoeff
        self.de_ctl_icoeff = de_ctl_icoeff
        self.de_ctl_dcoeff = de_ctl_dcoeff
        self.de_initial_step = de_initial_step
        self.de_adaptive = diffrax_adaptive
        self.de_split = de_split
        self.de_fake_pure_callback = diffrax_fake_pure_callback
        self.de_g_multiplier = diffrax_g_multiplier
        self.de_batch_channels = diffrax_batch_channels
        self.de_levy_area_approx = levy_areas[diffrax_levy_area_approx]
        self.de_error_order = diffrax_error_order
        self.de_sde_mode = diffrax_sde_mode
        self.de_g_reverse_time = diffrax_g_reverse_time
        self.de_g_time_scaling = diffrax_g_time_scaling
        self.de_g_split_time_mode = diffrax_g_split_time_mode

    # As slow and safe as possible.
    @staticmethod
    def t2j(t):
        return jax.block_until_ready(
            jax.numpy.array(numpy.array(t.detach().cpu().contiguous()))
        )

    @staticmethod
    def j2t(t):
        return torch.from_numpy(numpy.array(jax.block_until_ready(t))).contiguous()

    def check_solver_support(self):
        if not HAVE_DIFFRAX:
            raise RuntimeError(
                "Diffrax sampler requires diffrax and jax installed in venv."
            )

    def step(self, x):
        s, sn, sigma_down, sigma_up = self.de_get_step(x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x))
        ss = self.ss
        bidx = 0
        mcc = 0
        _b, c, h, w = x.shape
        interrupted = None
        t0, t1 = sigma_down.item(), s.item()

        def odefn_(t_orig, y_flat, args=()):
            nonlocal mcc, interrupted
            t = self.reverse_time(self.j2t(t_orig).to(s), t0, t1)
            if t <= 1e-05:
                return jax.numpy.zeros_like(y_flat)
            if mcc >= self.de_max_nfe:
                raise RuntimeError("DiffraxStep: Model call limit exceeded")
            y = self.j2t(y_flat.reshape(1, c, h, w)).to(x)
            t32 = t.to(s).clamp(min=1e-05)
            flat_shape = y_flat.shape
            del y_flat

            if not args and mcc == 0 and torch.all(t == s):
                mr_cached = True
                mr = ss.hcur
                mcc = 1
            else:
                mr_cached = False
                try:
                    if not args:
                        mr = self.call_model(y, t32, call_index=mcc, s_in=t.new_ones(1))
                    else:
                        print("TANGENTS")
                        mr = self.call_model(
                            y,
                            t32,
                            call_index=mcc,
                            tangents=args,
                            s_in=t.new_ones(1),
                        )
                except comfy.model_management.InterruptProcessingException as exc:
                    interrupted = exc
                    raise
                mcc += 1
            result = self.to_d(mr)[bidx if mr_cached else 0].reshape(*flat_shape)
            return self.t2j(-result)

        if not self.de_fake_pure_callback:

            def odefn(t, y_flat, args):
                return jax.experimental.io_callback(
                    odefn_, y_flat, t, y_flat, ordered=True
                )

        else:

            def odefn(t, y_flat, args):
                return jax.pure_callback(odefn_, y_flat, t, y_flat)

        def g(t, y, _args):
            if self.de_g_split_time_mode:
                val = jax.lax.cond(
                    t < t0 + (t1 - t0) * 0.5,
                    lambda: self.de_g_multiplier,
                    lambda: -self.de_g_multiplier,
                )
            else:
                val = self.de_g_multiplier
            if self.de_g_time_scaling:
                val *= self.reverse_time(t, t0, t1) if self.de_g_reverse_time else t
            if not self.de_batch_channels:
                return val
            return jax.numpy.float32(val).broadcast((y.shape[0],))

        def progress_callback():
            return f" ({mcc:>3}/{self.de_max_nfe:>3}) {self.de_solver_name}"

        term = diffrax.ODETerm(odefn)
        method = self.de_solver_method
        if self.de_adaptive:
            controller = diffrax.PIDController(
                atol=self.de_atol,
                rtol=self.de_rtol,
                dtmin=1e-05,
                pcoeff=self.de_ctl_pcoeff,
                icoeff=self.de_ctl_icoeff,
                dcoeff=self.de_ctl_dcoeff,
                error_order=self.de_error_order,
            )
        else:
            controller = diffrax.ConstantStepSize()

        if not self.de_adaptive:
            dt0 = (t1 - t0) / self.de_split
        else:
            dt0 = (t1 - t0) * self.de_initial_step
        if self.de_sde_mode:
            bm = diffrax.VirtualBrownianTree(
                t0=ss.sigmas.min().item(),
                t1=ss.sigmas.max().item(),
                tol=1e-06,
                levy_area=self.de_levy_area_approx,
                shape=(c,) if self.de_batch_channels else (),
                key=jax.random.PRNGKey(ss.noise.seed + ss.noise.seed_offset),
            )
            term = diffrax.MultiTerm(term, diffrax.ControlTerm(g, bm))
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
            if self.de_batch_channels:
                y_flat = x[bidx].flatten(start_dim=1)
            else:
                y_flat = x[bidx].unsqueeze(0).flatten(start_dim=1)
            y_flat = self.t2j(y_flat)
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                try:
                    solution = diffrax.diffeqsolve(
                        terms=term,
                        solver=method,
                        t0=t0,
                        t1=t1,
                        dt0=dt0,
                        y0=y_flat,
                        saveat=diffrax.SaveAt(t1=True),
                        stepsize_controller=controller,
                        progress_meter=StepCallbackTqdmProgressMeter(
                            step_callback=progress_callback,
                            refresh_steps=1,
                        ),
                    )
                except Exception:
                    if interrupted is not None:
                        raise interrupted
                    raise
            results.append(self.j2t(solution.ys).view(1, *x.shape[1:]))
            del solution
        result = torch.cat(results).to(x)
        sigma_up, result = yield from self.adjusted_step(sn, result, mcc, sigma_up)
        yield from self.result(result, sigma_up, sigma_down=sigma_down)
