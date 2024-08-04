# OCS Expressions

See [Filters](filter.md) for places where expressions apply.

## Expressions

OCS implements a simple expression language.

Supported math operators: `+`, `-`, `*`, `/`, `//` (integer division), `**` (power)

Supported logic operators: `||`, `&&`, `==`, `!=`, `>`, `<`, `>=`, `<=`

Operator precedence should generally work the way you'd expect.

You may surround a function name with backticks to turn it into a binary operator (only for functions that take two arguments).

Functions are called via `name(param1, param2)`. Keyword arguments may be passed using the `:>` operator. Example:
`name(param1, key :> 123, key2 :> otherfunction(10))`.

Symbols (simple string type) are defined using `'symbol_name` - note the solitary single quote. They may not contain spaces.

`;` can be used to sequence operations. I.E. `exp1 ; exp2` evaluates `exp1`, then `exp2` and then result of the expression is whatever `exp2` returned.

Like Python, a parenthesized expression with a trailing comma can be used to create an empty tuple. Example: `(1,)`

## Filter Variables

Indexes like `step` are zero-based: `0` will be the first step.

### Basic Variables

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

### Extended Variables

* `denoised`: From the current step or substep. May not be available in model `input` or group `pre_filter`.
* `cond`: From the current step or substep. May not be available in model `input` or group `pre_filter`.
* `uncond`: From the current step or substep. May not be available in model `input` or group `pre_filter`.
* `denoised_prev`: Only available when model history exists.
* `cond_prev`: Only available when model history exists.
* `cond_prev`: Only available when model history exists.

### Model Filter Variables

* `model_call`: Only applicable to `model` filters, will be the model call index. I.E. if the sampler calls the model three times, the filter would be called with model call indexes `0`, `1` and `2`.

Available in model filters, with the exception of the `input` filter.

* `denoised_curr`
* `cond_curr`
* `uncond_curr`

## Basic Expression Functions

 | | Name | Input | Output |
 | :--- | :--- | :--- | :--- |
 |⬤| `all` | `B`\* | `B` |
 | <td colspan=3 align=left>Evaluates to true if all its arguments evaluate to true. <br/> **Example:** `all(x > 1, y < 1)`</td> |
 |⬤| `any` | `B`\* | `B` |
 | <td colspan=3 align=left>Evaluates to true if any of its arguments evaluate to true. <br/> **Example:** `any(x > 1, y < 1)`</td> |
 |⬤| `between` | value:`N`, from:`N`, to:`N` | `B` |
 | <td colspan=3 align=left>Boolean range checking. <br/> **Example:** `between(value, low, high)`</td> |
 |⬤| `comment` | `*` | `null` |
 | <td colspan=3 align=left>Ignores any arguments passed to it (they won't be evaluated at all but must parse as a valid expression) and returns `None`</td> |
 |⬤| `dict` | `*`* | `dict` |
 | <td colspan=3 align=left>Constructs a dictionary from its keyword arguments. _Note_: You may not pass positional arguments. <br/> **Example:** `dict(key1 :> value1, keyN :> valueN)` |
 |⬤| `get` | name:`SY`, fallback:`*` | `*` |
 | <td colspan=3 align=left>Returns a variable if set, otherwise the fallback. <br/> **Example:** `get('somevar, 123)`</td> |
 |⬤| `if` | condition:`B`, then:`*`, else:`*` | `*` |
 | <td colspan=3 align=left>Conditional expressions. <br/> **Example:** `if(condition, true_expression, false_expression)`</td> |
 |⬤| `index` | index:`IDX`, value:`S \| T` | `*` |
 | <td colspan=3 align=left>Index function.</td> |
 |⬤| `is_set` | name:`SY` | `B` |
 | <td colspan=3 align=left>Tests whether a variable is set.</td> |
 |⬤| `max` | values:`SN` | `N` |
 | <td colspan=3 align=left>Maximum operation. _Note_: Takes one sequence argument. <br/> **Example:** `min((1, 2, 3))`</td> |
 |⬤| `min` | values: `SN` | `N` |
 | <td colspan=3 align=left>Minimum operation. _Note_: Takes one sequence argument. <br/> **Example:** `max((1, 2, 3))`</td> |
 |⬤| `mod` | lhs:`N`, rhs:`N` | `N` |
 | <td colspan=3 align=left>Modulus operation: <br/> **Example:** `mod(5, 2)`</td> |
 |⬤| `neg` | `N` | `N` |
 | <td colspan=3 align=left>Negation. <br/> **Example:** `neg(2)`</td> |
 |⬤| `not` | `B` | `B` |
 | <td colspan=3 align=left>Boolean negation</td> |
 |⬤| `s_` | start:`I(null)`, end:`I(null)`, step:`I(null)` | `slice` |
 | <td colspan=3 align=left>Creates a slice object from the `start`, `end`, `step` values. See Numpy [s_](https://numpy.org/doc/stable/reference/generatednumpy.s_.html)</td> |
 |⬤| `unsafe_call` | `callable`, `*`\* | `*` |
 | <td colspan=3 align=left>Allows calling an arbitrary callable. <br/> **Example:** `unsafe_call(some_callable, arg1, arg2, kwarg1 :> 123)`</td>

**Legend**: `B`=boolean, `N`=numeric, `NS`=scalar numeric, `I`=integer, `F`=float, `T`=tensor, `S`=sequence, `SN`=numeric sequence, `SY`=symbol, `*`=any -- parenthized values indicate argument defaults. `*` following the type indicates variable length arguments. For functions that take keyword arguments, the type will be written like "_name: `TYPE(default_value)`_".

## Tensor Expression Functions

*Tensor dimensions hint*: Most tensors you'll be dealing with are laid out as `batch`, `channels`, `height`, `width`. Negative indexes start from the end, so dimension `-1` would mean _width_ just the same as `3`.

 | | Name | Input | Output |
 | :--- | :--- | :--- | :--- |
 |⬤| `t_bleh_enhance` | tensor:`T`, mode:`SY`, scale:`N(1.0)` | `T`
 | <td colspan=3 align=left>Available if you have the [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) node pack installed. See [Filtering](filter.md#bleh_enhance). <br/> **Example:** `bleh_enhance(some_tensor, 'bandpass, 0.5)`</td> |
 |⬤| `t_blend` | tensor1:`T`, tensor2:`T`, scale:`N(0.5)`, mode:`SY(lerp)` | `T` |
 | <td colspan=3 align=left>Tensor blend operation. <br/> **Example:** `t_blend(t1, t2, 0.75, 'lerp)`</td> |
 |⬤| `t_contrast_adaptive_sharpening` | tensor:`T`, scale:`N(0.5)` | `T` |
 | <td colspan=3 align=left>Contrast adaptive sharpening. _Note_: Not recommended to call on noisy tensors (so `denoised` but probably not `x`). <br/> **Example:** `t_contrast_adaptive_sharpening(some_tensor, 0.1)`</td> |
 |⬤| `t_flip` | tensor:`T`, dim:`NS`, mirror:`B(false)` | `T` |
 | <td colspan=3 align=left>Flips a tensor on the specified dimension. If the third argument is true, it will be mirrored around the center in that dimension. <br/> **Example:** `t_flip(some_tensor, -1, true)`</td> |
 |⬤| `t_mean` | tensor:`T`, dim:`SN(-3, -2, -1)` | `T` |
 | <td colspan=3 align=left>Tensor mean, second argument is dimensions. <br/> **Example:** `t_mean(some_tensor, (-2, -1))`</td> |
 |⬤| `t_noise` | tensor:`T`, type:`SY(gaussian)` | `T` |
 | <td colspan=3 align=left>Generates un-normalized noise (use `t_norm` if you want to normalize it). If you have ComfyUI-sonar you can use any noise type that supports, otherwise only `gaussian`. The generated noise will have the same shape as the supplied tensor (hopefully, may not be true for every exotic noise type but at least should be broadcastable to the tensor). <br/> Example: `t_noise(some_tensor, 'pyramid)`</td> |
 |⬤| `t_norm` | tensor:`T`, factor:`N(1.0)`, dim:`SN(-3, -2, -1)` | `T` |
 | <td colspan=3 align=left>Tensor normalization (subtracts mean, divides by std). <br/> **Example:** `t_norm(some_tensor, 1.0, (-2, -1))`</td> |
 |⬤| `t_sonar_power_filter` | tensor:`T`, filter:`dict` | `T` |
 | <td colspan=3 align=left>Available if you have [ComfyUI-sonar](https://github.com/blepping/ComfyUI-sonar) installed. See [Filtering](filter.md#sonar_power_filter). Constructs a power filter from a dictionary argument. _Note_: May be slow as the filter is reconstructed on every evaluation. <br/> **Example:** `t_sonar_power_filter(some_tensor, dict(alpha :> 0.1, min_freq :> 0.2, max_freq :> 0.6))`</td> |
 |⬤| `t_roll` | tensor:`T`, amount:`NS(0.5)`, dim:`SN((-2,))` | `T` |
 | <td colspan=3 align=left>Rolls a tensor along the specified dimensions. If amount is >= -1.0 and < 1.0 this will be interpreted as a percentage. <br/> **Example:** `t_roll(some_tensor, 10, (-2,))`</td> |
 |⬤| `t_scale` | tensor:`T`, scale:`SN \| NS`, mode:`SY(bicubic)`, absolute_scale:`B(false)` | `T` |
 | <td colspan=3 align=left>Scales a tensor. If scale is a tuple, it will be interpreted as `(height, width)`. When `absolute_scale` is not set, the scales will be interpreted as percentages otherwise absolute values will be used. <br/> Example: `t_scale(some_tensor, (0.75, 0.5), 'bilinear)`</td> |
 |⬤| `t_std` | tensor:`T`, dim:`SN(-3, -2, -1)` | `T` |
 | <td colspan=3 align=left>Tensor std, second argument is dimensions. <br/> **Example:** `t_std(some_tensor, (-2, -1))`</td> |
 |⬤| `unsafe_tensor_method` | `T`, `SY`, `*`\* | `*` |
 | <td colspan=3 align=left>Unsafe tensor method call. See note below. <br/> **Example:** `unsafe_tensor_method(some_tensor, 'mul, 10)`</td> |
 |⬤| `unsafe_torch` | path:`SY` | `*` |
 | <td colspan=3 align=left>Unsafe Torch module attribute access. See note below. <br/> **Example:** `unsafe_torch('nn.functional.interpolate)`</td> |

**Note on `unsafe_tensor_method` and `unsafe_torch`**: These functions are disabled by default. If the environment variable `COMFYUI_OCS_ALLOW_UNSAFE_EXPRESSIONS` is set to anything then you can use `unsafe_tensor_method` with a whitelisted set of methods (best effort to avoid anything actually unsafe). If the environment variable `COMFYUI_OCS_ALLOW_ALL_UNSAFE` is set to anything then `unsafe_torch` is enabled and `unsafe_tensor_method` will allow calling any method. ***WARNING***: Allowing _all_ unsafe with workflows you don't trust is _not_ recommended and a malicious workflow will likely have access to anything ComfyUI can access. It is effectively the same as letting the workflow run an arbitrary script on your system.
