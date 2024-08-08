# OCS Filters

Filters allow changing sampler inputs/outputs, model input/outputs, generated noise and so on. They are
configured by the advanced YAML/JSON parameter block in the node.

## Filter Support

### `OCS Substeps`

Set via the `pre_filter` and `post_filter` keys.

### `OCS Group`

Set via the `pre_filter` and `post_filter` keys.

*Note*: Since the group pre-filter may be called before any model calls, variables like `denoised` may not be available
in expressions.

### `OCS Sampler`

**Noise**

```yaml
noise:
    # Or set to a valid filter definition.
    filter: null
```

**Model**

*Note*: Since the model filters may be called before any other model calls, variables like `denoised` may not be available
in expressions. With the exception of the `input` filter you will have access to `denoised_curr`, `cond_curr`, etc.
See [Expressions](expression.md#model-filter-variables).

```yaml
model:
    filter:
        # Applies to the input passed to the model.
        input: null

        # Applies to denoised output.
        denoised: null

        # Applies to JVP denoised output (only used by TTM sampler)
        jdenoised: null

        # Applies to cond output.
        cond: null

        # Applies to uncond output.
        uncond: null

```

### Immiscible

`immiscible` is a special type of filter: in this case, you do not set `filter_type`. Normal filter keys
apply in the places where `immiscible` can be set.

## Filter Definitions

For information about expressions, see [Expressions](expression.md).

A basic filter supports these keys:

```yaml
enabled: true

filter_type: simple

# Expression that is evaluated to determine whether the filter applies. May be null.
# If set, should evaluate to a boolean.
when: null

blend_mode: lerp

# Blend strength applied to output.
strength: 1.0

# Input expression. input, ref, output and final should evaluate to a tensor.
input: default

# Reference expression (only used for immiscible noise currently).
ref: default

# Output expression.
output: default

# Final expression - occurs *after* blending.
final: default
```

There may be additional keys depending on the filter type.

If you have [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) available, you can use any blend mode it supports. Otherwise OCS provides these built-in blend modes: `lerp`, `a_only`, `b_only`. _Note_: `a` is considered the original value, `b` the changed value. `a_only` and `b_only` will still scale their output by the `strength`.

## Filter Types

### `simple`

Base filter, no special behavior. No additional parameters.

### `blend`

Blends the result of two other filters.

Keys:

```yaml
# No default, value for example purpose only.
filter1:
    filter_type: simple

# No default, value for example purpose only.
filter2:
    filter_type: simple
```

### `list`

A list of filters. The output of the previous is given to the next as input.
The `list` filter's blend applies to the output from the final filter in the list.

Keys:

```yaml
# Values for example only, default is an empty list.
filters:
    - filter_type: simple
      strength: 1.0
    - filter_type: simple
      strength: 1.0
```

### `bleh_enhance`

Available if you have the [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) node pack installed. See:
https://github.com/blepping/ComfyUI-bleh#enhancement-types

Keys:

```yaml
enhance_mode: null
enhance_scale: 1.0
```

### `bleh_ops`

Available if you have the [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) node pack installed. See:
https://github.com/blepping/ComfyUI-bleh#blehblockops

Keys:

```yaml
# May be specified as a string containing the YAML rule definitions or inline.
# Values for example only, default is an empty list of ops.
ops:
    - if:
        to_percent: 0.5
      ops: # Not recommended to actually do this.
        - [flip, { direction: h }]
        - [roll, { direction: channels, amount: -2 }]
```

### `sonar_power_filter`

Available if you have [ComfyUI-sonar](https://github.com/blepping/ComfyUI-sonar) installed. See:
https://github.com/blepping/ComfyUI-sonar/blob/main/docs/advanced_power_noise.md

Keys:

```yaml
power_filter:
    mix: 1.0
    normalization_factor: 1.0
    common_mode: 0.0
    channel_correlation: "1,1,1,1,1,1"
    alpha: 0.0
    min_freq: 0.0
    max_freq: 0.7071
    stretch: 1.0
    rotate: 0.0
    pnorm: 2.0
    scale: 1.0
    compose_mode: max

    # If specified should be another power filter definition.
    compose_with: null
```
