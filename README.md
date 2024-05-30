# Overly Complicated Sampling
Wildly unsound and experimental sampling for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Description

Very unstable, experimental and mathematically unsound sampling for ComfyUI.

Current status: In flux, not suitable for general use.

## Nodes

### ComposableSampler

**Possible Parameters**

* `avgmerge_stretch`(`0.4`): Used for `average` and `sample` merge types. See below.
* `model_call_cache`(unset): Caches the result of model calls at n+1 (where `n` is the number of model evaluations per step). For example, Bogacki is three model calls per step: whether the first one runs is dependent on the merge strategy. After that, Bogacki calls the model two more times. If you set `model_call_cache` to `1` then the result of that second call will be cached and if you're running two Bogacki substeps then the second one will use the cached version. Massively accelerates inference (especially when using the `average` merge strategy) but is likely very unsound and inaccurate.

#### Merging

When running multiple substeps per step, the results will combined based on the merge strategy. Possible strategies:

* `normal`: The model is called at least once per substep (and possibly additional times for higher order samplers). The result of each substep is noised and the next substep uses that result. Then all the results are averaged.
* `divide`: Creates a linear schedule between the current sigma and the next and runs the substeps in sequence. The model is called at least once per substep.
* `average`: The model is called once at the beginning of the step and substeps share that result (but it may be called additional times for higher order samplers). This means substeps for samplers like reversible Euler, Heun 1s, DPM++ 2m SDE are essentially free. May be theoretically very unsound and inaccurate, requires manual tweaking of settings like `s_noise`. Supports the parameter `avgmerge_stretch`(`0.4`) which basically rolls back the current sigma and adds some noise (otherwise running a substep is deterministic and there would be no point to running a sampler like Euler more than once).
* `sample`: Like `average` (and uses `avgmerge_stretch`) but instead of simply using the average, it does a sampler step toward it instead. You can plug in any substep sampler to the `merge_sampler_opt` input (if unconnected and the merge method is `sample` then Euler will be used).

### ComposableStepSampler

This node has a text input for YAML (or JSON) advanced parameters.

For example, you could enter something like this in the field:

```yaml
reta: 1.1
leap: 3
dyn_deta_mode: "deta"
```

**Possible Parameters**

#### General

* `eta`(`1.0`): Will override `eta` in the node if set.
* `s_noise`(`1.0`): Will override `s_noise` in the node if set.
* `solver_type`(`midpoint`): Applies to DPM++ 2m SDE. May be one of `midpoint` or `heun` (`midpoint` is generally recommended).

#### Reversible

* `reta`(`1.0`): Reverse ETA.

#### Dancing

* `leap`(`2`): Distance to try to leap forward. If you set `leap` to `1` you just get plain old Euler ancestral.
* `deta`(`1.0`): ETA used for dance steps.
* `dyn_deta_start`(`unset`) and `dyn_deta_end`(`unset`): No effect unless both values are set. Will interpolate between start and end based on the percentage of sampling.
* `dyn_deta_mode`(`lerp`): May be one of:
    * `deta`: Scales `deta` based on the value from `dyn_deta_start/end`.
    * `lerp`: Does the dance step according to `deta` and then LERPs the non-dance sample result with the dance sample result based on the scale calculated from `dyn_deta_start/end` (which is `1.0` if they are unset). For example, if the dance scale is `0.5` you will get 50% normal sampling, 50% dancing sampling.

#### RES

* `res_simple_phi`(`false`): Applies to RES. Uses a faster but possibly less accurate method for calculating phi. What does phi do? I haven't the foggiest!
* `res_c2`(`0.5`): Applies to RES. Solver partial step size, the default of `0.5` appears to use the midpoint. Setting it to a lower value might possibly be more accurate but slower?

## Credits

I can move code around but sampling math and creating samplers is far beyond my ability. I didn't write any of the original samplers:

* Euler, DPM++ 2m, 2m SDE and 3m SDE samplers based on ComfyUI's implementation.
* Reversible Heun, Reversible Heun 1s, RES, Trapezoidal, Bogacki, Reversible Bogacki, RK4 and Euler Dancing samplers based on implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
* Normal substep merge strategy based on implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers

Thanks!
