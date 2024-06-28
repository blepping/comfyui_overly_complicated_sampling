# Overly Complicated Sampling
Wildly unsound and experimental sampling for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Description

Very unstable, experimental and mathematically unsound sampling for ComfyUI.

Current status: In flux, not suitable for general use.

*Note*: You will basically always have to tweak settings like `s_noise` to get a good result. If the generation looks smooth/undetailed increase `s_noise` somewhere. If it looks crunchy, super high contrast, etc then try reducing noise.

## Usage

First, a note on the basic structure:

![Basic nodes example](assets/basic_sampling.png)

The sampler node connects to a group node. You can chain group nodes, however only one can match per step. Checking for a match starts at the group furthest from the sampler node. So if you have `Group1 -> Group2 -> Group3 -> Sampler`, the groups will be tried starting from `Group1`. Currently matching groups is only time based.

You will then connect a substeps node to the group. These can also be chained and like groups, execution starts with the node furthest from the group. I.E.: `Substeps1 -> Substeps2 -> Substeps3 -> Group` will start with `Substeps1`.

Most of these nodes have a text parameter input (YAML format - JSON is also valid YAML so you can use that instead if you prefer) and a parameter input. The parameter input can be used to specify stuff like custom noise types.

## Nodes

### `OCS Sampler`

The main sampler node, with an output suitable for connecting to a `SamplerCustom`. This node has builtin support for Restart sampling, if you are
using Restart don't use the `RestartSampler` node.

You can connect a chain of `OCS Group` nodes to it and it will choose one per step (based on conditions like time).

#### Input Parameters

* `restart_custom_noise`: Value type: `SONAR_CUSTOM_NOISE`. Allows specifying a custom noise type when used with Restart sampling.

#### Text Parameters

Shown in YAML with default values.

```yaml
# Noise scale. May not do anything currently.
s_noise: 1.0

# ETA (basically ancestralness). May not do anything currently.
eta: 1.0

# Reversible ETA (used for reversible samplers). May not do anything currently.
reta: 1.0

# Scales the noise added when used with Restart sampling.
restart_s_noise: 1.0


# The noise block allows defining global noise sampling parameters.
noise:
    # You can disable this to allow GPU noise generation. I believe it only makes a difference for Brownian.
    cpu_noise: true

    # ComfyUI has a bug where if you disable add_noise in the sampler, no seed gets set. If you
    # are manually noising a sample and have add_noise turned off then you should enable this if
    # you want reproducible generations.
    set_seed: false

    # Global scale scale for generated noise
    scale: 1.0

    # Whether the generated noise should be normalized before use. Generally a good idea to leave enabled.
    normalize_noise: true

    # Dimensions to normalize over (when normalization is enabled). Negative values mean starting
    # from the end (i.e. -1 means the last dimension, -2 means the penultimate dimension).
    # Latents generally have these dimensions: batch, channels, height, width
    # The default of [-3, -2, -1] normalizes noise over the batch. You can try something like
    # [-2, -1] to normalize over the batch and channels. See: https://pytorch.org/docs/stable/generated/torch.std.html
    normalize_dims: [-3, -2, -1]

    # When caching, the batch size for chunks of noise to generate in advance. Generating a batch of noise
    # can be more efficient than generating on demand when using a high number of substeps (>10) per step.
    batch_size: 32

    # Whether to cache noise.
    caching: true

    # Interval (in full steps) to reset the cache. Brownian noise takes time into account so
    # if using Brownian you will generally want to reset each step.
    cache_reset_interval: 1


# Model calls can be cached. This is very experimental: I don't recommend using it
# unless you know what you're doing.
model_call_cache:

    # The cache size.
    size: 0

    # Threshold for model call caching. For example if you have size=3 and threshold=1
    # then model calls 1 through 3 will be cached, but model call 0 will not be (the first one).
    # Additional explanation: Some samplers call the model multiple times per step. For example,
    # Bogacki uses three model calls: 0, 1, 2
    threshold: 1

    # Maximum use count for cache items.
    max_use: 1000000
```

Any parameters you don't specify will use the defaults. For example if your text parameter block is:

```yaml
noise:
    cpu_noise: false
```

Then the rest of the parameters will use the defaults shown above.

### `OCS Group`

Defines a group of substeps.


#### Merging

When running multiple substeps per step, the results will combined based on the merge strategy. Possible strategies (in order of least weird to most weird):

* `simple`: Doesn't merge anything: only runs a single substep per step.
* `divide`: Creates a linear schedule between the current sigma and the next and runs the substeps in sequence. The model is called at least once per substep.
* `normal`: The model is called at least once per substep (and possibly additional times for higher order samplers). Each substep shares the first model call result. The results are averaged together. *Note*: Since the first model call is shared and the initial input is the same for each substep, there is no point in running multiple identical substeps. Also note: This merge strategy doesn't work well with non-ancestral samplers (i.e. dpmpp_2m or any sampler with `eta: 0`).
<!--
* `average`: The model is called once at the beginning of the step and substeps share that result (but it may be called additional times for higher order samplers). This means substeps for samplers like reversible Euler, Heun 1s, DPM++ 2m SDE are essentially free. May be theoretically very unsound and inaccurate, requires manual tweaking of settings like `s_noise`. Supports the parameter `avgmerge_stretch`(`0.4`) which basically rolls back the current sigma and adds some noise (otherwise running a substep is deterministic and there would be no point to running a sampler like Euler more than once).
* `sample`: Like `average` (and uses `avgmerge_stretch`) but instead of simply using the average, it does a sampler step toward that instead. You can plug in any substep sampler to the `merge_sampler_opt` input (if unconnected and the merge method is `sample` then Euler will be used). *Note*: Substeps in the attached sampler will be ignored.
* `sample_uncached`: Similar to `sample`, however it calls the model per substep instead of caching the result and sharing it. Aside from sampling toward the result, it works more like the `normal` merge strategy. Theoretically it should be better because it's taking less shortcuts but results seem worse.

When using `average` and `sample` merge strategies and with model call caching enabled you can get away with setting substeps super high. Running something like 100 substeps is actually quite practical and seems to work well.
-->

#### Node Parameters

* `merge_method`: One of `simple`, `divide`, `normal` <!--, `average`, `sample`, `sample_uncached`. -->
* `time_mode`(`step`): One of `step`, `step_pct`, `sigma`. Time matching mode. Matching based on steps generally will be simplest. Matches are inclusive and steps start at 0 (so step 0 is the first step). `step_pct` is the percentage of total steps (1.0=100%, 0.5=50%, etc).
* `time_start`(`0`): Match start time.
* `time_end`(`999`): Match end time.

Example:

![Group time filter example](assets/group_time_example.png)

The left side group matches steps 0, 1, 2. The right side group matches all steps. This setup will use whatever substeps are connected to the first group for the first three steps and the second group will handle the rest.


#### Input Parameters

Currently unused for groups.

<!--
* `merge_sampler`: Value type: `OCS_SUBSTEPS`. Only used when `merge_method` is `sample` or `sample_uncached`. Allows defining the sampler used for merging substeps.
-->


#### Text Parameters

Shown in YAML with default values.

```yaml
# Noise scale. May not do anything currently.
s_noise: 1.0

# ETA (basically ancestralness). May not do anything currently.
eta: 1.0

# Reversible ETA (used for reversible samplers). May not do anything currently.
reta: 1.0

# Currently unused.
avgmerge_stretch: 0.4
```

### `OCS Substeps`

#### Step Methods (Samplers)

In alphabetical order.

* `bogacki`: CFG++ enabled.
* `deis`: May not work work ETA > 0. See parameters: `history_limit`
* `dpmpp_2m_sde`: See parameters: `history_limit`
* `dpmpp_2m`: `eta` and `s_noise` parameters are ignored. See parameters: `history_limit`
* `dpmpp_2s`
* `dpmpp_3m_sde`: See parameters: `history_limit`
* `dpmpp_sde`
* `euler_cycle`: CFG++ enabled. See parameters: `cycle_pct`
* `euler_dancing`: Pretty broken currently, will probably require increased `s_noise` values. See parameters: `deta`, `leap`, `deta_mode`
* `euler`: CFG++ enabled.
* `heunpp`: May not work work ETA > 0. See parameters: `max_order`
* `ipndm_v`: May not work work ETA > 0. See parameters: `history_limit`
* `ipndm`: May not work work ETA > 0. See parameters: `history_limit`
* `res`
* `reversible_bogacki`: CFG++ enabled.
* `reversible_heun`: CFG++ enabled.
* `reversible_heun_1s`: CFG++ enabled. See parameters: `history_limit`
* `rk4`: CFG++ enabled.
* `tde`: Uses the [torchdiffeq](https://github.com/rtqichen/torchdiffeq) ODE backend. See `ode_*` parameters below. CFG++ enabled.
* `tode`: Uses the [torchode]((https://github.com/martenlienen/torchode)) ODE backend. See `ode_*` parameters below. CFG++ enabled.
* `trapezoidal`: CFG++ enabled.
* `trapezoidal_cycle`: CFG++ enabled. See parameters: `cycle_pct`
* `ttm_jvp`: TTM is a weird sampler. If you're using model caching you must make sure the entries TTM uses are populated first (by having it run before any other samplers that call the model multiple times). It may also not work with some other model patches and upscale methods. See parameters: `alternate_phi_2_calc`

**ODE Solvers** (`tde`, `tode`):

You will need to have the relevant Python package installed in your venv to use these. TDE cannot handle batches and
each batch item will be evaluated separately. Using `tode` may be faster for batch sizes over 1.

`ode_solver` types for TDE: adaptive: `dopri8`, `dopri5`, `bosh3`, `fehlberg2`, `adaptive_heun`, fixed step: `euler`, `midpoint`, `rk4`, `explicit_adams`, `implicit_adams`

`ode_solver` types for TODE: adaptive only: `dopri5`, `tsit5`, `heun`. I haven't much luck with anything other than `dopri5`.

Note that adaptive solvers may be _very_ slow. Think along the lines of 20-100 model calls per substep (or in other words, the equivalent for running that many `euler` steps). Tolerances only apply to adaptive solvers.

**Cycle Samplers** (`euler_cycle`, `trapezoidal_cycle`)

Basically a different approach to ancestral sampling. First a crash course on how sampling works:

Each step has an expected noise level, with the first step generally being pure noise and the end of the last step aiming to end with no noise remaining. Let's say the image on the current step is called `x`, calling the model with `x` gives us a prediction of what the image looks like with all noise removed (`denoised`), however the model is not capable of just removing all the noise in a single step: its prediction will be imprecise. `x - denoised` leaves us with just the noise (we subtract the prediction which theoretically has no noise from the noisy sample). This is a very simplified, but the idea is basically to add the noise back into `denoised`, but scaled so that it matches the amount of noise expected on the _next_ step. `denoised + noise * expected_noise_at_next_step`.

When doing ancestral sampling, we actually _overshoot_ expected noise for the next step and add less than that amount back to `denoised`. Then we generate some of our own noise and add it, scaled so that the result matches `expected_noise_at_next_step`. `eta` controls how the scale of the overshoot.

The difference with cycle is that instead of adding `noise * expected_noise_at_next_step`, we instead first add `noise * (expected_noise_at_next_step * (1.0 - cycle_pct))` and then we generate noise and scale it to `cycle_pct` and add it too. Just for example, suppose `cycle_pct` is `0.2`: we'll add 80% of the expected noise at the next step (`1.0 - 0.2 == 0.8`) and then generate the remaining 20% and add it in to meet the expected amount. I don't recommend setting `cycle_pct` to values over `0.5`, especially if using "weird" noise types.

#### Node Parameters

* `substeps`(`1`): Number of substeps. Generally involves a model call per substep, so for example setting this to 4 would approximately quadruple sampling time.
* `step_method`(`euler`): Method used for sampling the substeps. May include a parenthesized number (i.e. `rk4 (3)`) which denotes the number of _extra_ model calls required per sample. At least one is always required. So `euler` requires 1 in total, `rk4` requires 4 in total. RK4 is about 4 times slower than `euler`.

#### Input Parameters

* `custom_noise`: Value type: `SONAR_CUSTOM_NOISE`. Allows specifying a custom noise type for samplers that generate noise (most of them).

#### Text Parameters

Shown in YAML with default values.

```yaml
# Scale for added noise.
s_noise: 1.0

# ETA (basically ancestralness).
eta: 1.0
# No effect unless both start and end are set. Will scale the eta value based on the
# percentage of sampling. In other words, eta*dyn_eta_start at the beginning,
# eta*dyn_eta_end at the end.
dyn_eta_start: null
dyn_eta_end: null

# CFG++ scale (see https://cfgpp-diffusion.github.io/)
# Setting this to 1.0 is the equivalent of enabling it. Can also be set
# to a negative value (I don't recommend going lower than -0.5).
cfgpp_scale: 0

### Reversible Settings ###

# Reversible ETA (used for reversible samplers).
reta: 1.0
# Scale of the reversible correction. Can also be set to a negative value.
reversible_scale: 1.0
# No effect unless both start and end are set. Will scale the reta value based on the
# percentage of sampling. In other words, reta*dyn_reta_start at the beginning,
# reta*dyn_reta_end at the end.
dyn_reta_start: null
dyn_reta_end: null

### ODE Sampler Settings ###

# Solver type.
ode_solver: dopri5
# Relative tolerance (log 10)
ode_rtol: -1.5
# Absolute tolerance (log 10)
ode_atol: -3.5
# Max model calls allowed to compute the solution. If the limit is exceeded, it is an error.
ode_max_nfe: 1000
# Hack that seems to help results. Set to 0 to disable.
ode_fixup_hack: 0.025

## torchdiffeq (tde) specific parameters ##
# Used to split the step into sections. Useful for fixed step methods.
ode_split: 1

## torchode (tode) specific parameters ##
# Initial step size (as a percentage).
ode_initial_step: 0.25

# Coefficients for the step size PID controller.
# See https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller
# These values seem okay with dopri5.
ode_ctl_pcoeff: 0.3
ode_ctl_icoeff: 0.9
ode_ctl_dcoeff: 0.2

# Controls whether to compile the solver. May or may not work,
# also may or may not be a speed increase as the compiled solver is
# not cached between substeps.
ode_compile: false

### Other Sampler Specific Parameters ###

# Used for some samplers that use history from previous steps.
# List of samplers and default value below:
#   dpmpp_2m: 1
#   dpmpp_2m_sde: 1
#   dpmpp_3m_sde: 2
#   reversible_heun_1s: 1
#   ipndm: 3
#   ipndm_v: 3
#   deis: 2 (max 3)
history_limit: 999 # Varies based on sampler.

# Used for some samplers with variable order. List of samplers and default value below:
#   heunpp2: 3
max_order: 999 # Varies based on sampler.

# Used for dpmpp_2m. One of midpoint, heun
solver_type: "midpoint"

# Coefficients mode for DEIS. One of tab or rhoab.
deis_mode: "tab"

# Used for samplers with cycle in the name. Controls how much noise is cycled per step.
cycle_pct: 0.25

# Used for ttm_jvp. Supposed works better when ETA > 0
alternate_phi_2_calc: true

# Parameters for dancing samplers:
# Number of steps to leap ahead.
leap: 2
# ETA for dance steps
deta: 1.0
# dyn_deta works the same as dyn_eta/reta. See above.
dyn_deta_start: null
dyn_deta_end: null
# One of lerp, lerp_alt, deta
dyn_deta_mode: "lerp"
```


## Credits

I can move code around but sampling math and creating samplers is far beyond my ability. I didn't write any of the original samplers:

* Euler, Heun++2, DPMPP SDE, DPMPP 2S, DPM++ 2m, 2m SDE and 3m SDE samplers based on ComfyUI's implementation.
* Reversible Heun, Reversible Heun 1s, RES, Trapezoidal, Bogacki, Reversible Bogacki, RK4 and Euler Dancing samplers based on implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
* TTM JVP sampler based on implementation written by Katherine Crowson (but yoinked from the Extra-Samplers repo mentioned above).
* IPNDM, IPNDM_V and DEIS adapted from https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py (I used the Comfy version as a reference).
* Normal substep merge strategy based on implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers

This repo wouldn't be possible without building on the work of others. Thanks!
