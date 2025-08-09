SAMPLER_LIST = []

STEP_SAMPLERS = {}
STEP_SAMPLER_SIMPLE_NAMES = {}


def add(*objs):
    global SAMPLER_LIST
    SAMPLER_LIST += objs


def init():
    global STEP_SAMPLERS, STEP_SAMPLER_SIMPLE_NAMES
    STEP_SAMPLER_SIMPLE_NAMES.clear()
    STEP_SAMPLERS.clear()
    euler = None
    temp = []
    for c in SAMPLER_LIST:
        mc = c.model_calls
        if mc == 0:
            prettymc = ""
        elif isinstance(mc, tuple):
            prettymc = f" ({mc[0]}-{mc[-1]})"
        elif mc < 0:
            prettymc = " (variable)"
        else:
            prettymc = f" ({mc})"
        if c.name == "euler":
            euler = c
        temp.append((f"{c.name}{prettymc}", c))
    temp.sort(key=lambda item: item[1].name)
    if euler is None:
        raise RuntimeError(
            "Impossible: euler sampler not found when building sampler registry"
        )
    STEP_SAMPLERS["default (euler)"] = euler
    STEP_SAMPLERS |= {k: v for k, v in temp}
    STEP_SAMPLER_SIMPLE_NAMES["default"] = euler
    STEP_SAMPLER_SIMPLE_NAMES |= {v.name: v for _k, v in temp}
