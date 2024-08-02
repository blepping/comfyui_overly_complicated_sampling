# OCS Expressions

See [Filters](filter.md) for places where expressions apply.

## Expressions

OCS implements a simple expression language.

Supported math operators: `+`, `-`, `*`, `/`, `//`, `**`

Supported logic operators: `||`, `&&`, `==`, `!=`, `>`, `<`, `>=`, `<=`

Operator precedence should generally work the way you'd expect.

You may surround a function name with backticks to turn it into a binary operator (only for functions that take two arguments).

Functions are called via `name(param1, param2)`. Keyword arguments may be passed using the `:>` operator. Example:
`name(param1, key :> 123, key2 :> otherfunction(10))`.

Symbols (simple string type) are defined using `'symbol_name` - note the solitary single quote. They may not contain spaces.

## Filter Variables

Indexes like `step` are zero-based: `0` will be the first step.

**Basic Variables**

* `default`: Context specific default value. i.e. if used in an `input` expression this would be `x`, if used for `output` this would be the current result.
* `step`: Current step.
* `substep`: Current substep.
* `dt`: `sigma_next - sigma`
* `sigma_idx`: Index of the current sigma. Note that when using restarts this will be based on the restart sigma chunks, not full sigma list.
* `sigma`: The current sigma.
* `sigma_next`: The next sigma.
* `sigma_down`: The down sigma in ancestral sampling.
* `sigma_up`: The up sigma in ancestral sampling.
* `sigma_prev`: The previous sigma (may be `None`).
* `hist_len`: Current available history length. "Now" counts as one.
* `sigma_min`: The minimum sigma (based on the full list).
* `sigma_max`: The maximum sigma (based on the full list).
* `step_pct`: Percentage for the current step (based on total steps).
* `total_steps`: Total steps to be sampled.

**Extended Variables**

* `denoised`: From the current step or substep. May not be available in model `input` or group `pre_filter`.
* `cond`: From the current step or substep. May not be available in model `input` or group `pre_filter`.
* `uncond`: From the current step or substep. May not be available in model `input` or group `pre_filter`.
* `denoised_prev`: Only available when model history exists.
* `cond_prev`: Only available when model history exists.
* `cond_prev`: Only available when model history exists.
* `model_call`: Only applicable to `model` filters, will be the model call index. I.E. if the sampler calls the model three times, the filter would be called with model call indexes `0`, `1` and `2`.
* `x`

**Model Output Expressions**

Only available in model output expressions (after a model result is available).

* `denoised_curr`
* `cond_curr`
* `uncond_curr`

## Basic Expression Functions

|Name|Args|Result|Description|
|-|-|-|-|
|`not`|`B`|`B`|Boolean negation|
|`mod`|`N`, `N`|`N`|Modulus operation: Example: `mod(5, 2)`|
|`neg`|`N`|`N`|Negation. Example: `neg(2)`|
|`between`|`N`, `N`, `N`|`B`|Boolean range checking. Example: `between(value, low, high)`|
|`if`|`B`, `*`, `*`|`*`|Conditional expressions. Example: `if(condition, true_expression, false_expression)`|
|`min`|`N`*|`N`|Minimum operation.|
|`max`|`N`*|`N`|Maximum operation.|
|`**`|`N`, `N`|`N`|Power operator.|
|`//`|`N`, `N`|`N`|Integer division operator.|
|`is_set`|`SY`|`B`|Tests whether a variable is set.|
|`get`|`SY`, `*`|`*`|Returns a variable if set, otherwise the fallback. Example: `get('somevar, 123)`|
|`index`|`IDX`, `S` \| `T`|`*`|Index function.|
|`s_`|`I(null)`, `I(null)`, `I(null)`|`slice`|Creates a slice object from the `start`, `end`, `step` values. See Numpy [s_](https://numpy.org/doc/stable/reference/generated/numpy.s_.html)|
|`unsafe_call`|`callable`, `*`\*|`*`|Allows calling an arbitrary callable. Example: `unsafe_call(some_callable, arg1, arg2, kwarg1 :> 123)`|

**Legend**: `B`=boolean, `N`=numeric, `NS`=scalar numeric, `I`=integer, `F`=float, `T`=tensor, `S`=sequence, `SN`=numeric sequence, `SY`=symbol, `*`=any -- parenthized values indicate argument defaults. `*` following the type indicates variable length arguments.

## Extended Expression Functions

|Name|Args|Result|Description|
|-|-|-|-|
|`t_norm`|`T`,`N(1.0)`, `SN(-3, -2, -1)`|`T`|Tensor normalization (subtracts mean, divides by std). Example: `t_norm(some_tensor, 1.0, (-2, -1))`|
|`t_mean`|`T`, `SN(-3, -2, -1)`|`T`|Tensor mean, second argument is dimensions. Example: `t_mean(some_tensor, (-2, -1))`|
|`t_std`|`T`, `SN(-3, -2, -1)`|`T`|Tensor std, second argument is dimensions. Example: `t_std(some_tensor, (-2, -1))`|
|`t_blend`|`T`, `T`, `N(0.5)`, `SY(lerp)`|Tensor blend operation. Example: `t_blend(t1, t2, 0.75, 'lerp)`|
|`unsafe_tensor_method`|`T`, `SY`, `*`\*|`*`|Unsafe tensor method call. See note below. Example: `unsafe_tensor_method(some_tensor, 'mul, 10)`|
|`unsafe_torch`|`SY`|`*`|Unsafe Torch module attribute access. See note below. Example: `unsafe_torch('nn.functional.interpolate)`|
|`bleh_enhance`|`T`, `SY`, `N(1.0)`|`T`|Available if you have the [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) node pack installed. See [Filtering](filter.md#bleh_enhance).|

**Note on `unsafe_tensor_method` and `unsafe_torch`**: These functions are disabled by default. If the environment variable `COMFYUI_OCS_ALLOW_UNSAFE_EXPRESSIONS` is set to anything then you can use `unsafe_tensor_method` with a whitelisted set of methods (best effort to avoid anything actually unsafe). If the environment variable `COMFYUI_OCS_ALLOW_ALL_UNSAFE` is set to anything then `unsafe_torch` is enabled and `unsafe_tensor_method` will allow calling any method. ***WARNING***: Allowing _all_ unsafe with workflows you don't trust is _not_ recommended and a malicious workflow will likely have access to anything ComfyUI can access. It is effectively the same as letting the workflow run an arbitrary script on your system.
