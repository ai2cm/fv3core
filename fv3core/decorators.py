import collections
import functools
import types
import numpy
import gt4py as gt
import fv3core
import numpy
import fv3core._config as spec
from fv3gfs.util import Quantity
import xarray as xr


ArgSpec = collections.namedtuple(
    "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
)
VALID_INTENTS = ["in", "out", "inout", "unknown"]


def state_inputs(*arg_specs):
    for aspec in arg_specs:
        if aspec.intent not in VALID_INTENTS:
            raise ValueError(
                f"intent for {aspec.arg_name} is {aspec.intent}, must be one of {VALID_INTENTS}"
            )

    def decorator(func):
        @functools.wraps(func)
        def wrapped(state, *args, **kwargs):
            namespace_kwargs = {}
            for aspec in arg_specs:
                arg_name, standard_name, units, intent = aspec
                if standard_name not in state:
                    raise ValueError(f"{standard_name} not present in state")
                elif units != state[standard_name].units:
                    raise ValueError(
                        f"{standard_name} has units {state[standard_name].units} when {units} is required"
                    )
                else:
                    namespace_kwargs[arg_name] = state[standard_name].storage
                    namespace_kwargs[arg_name + "_quantity"] = state[standard_name]
            func(types.SimpleNamespace(**namespace_kwargs), *args, **kwargs)

        return wrapped

    return decorator
