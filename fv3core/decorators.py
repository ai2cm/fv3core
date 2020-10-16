import collections
import functools
import types
from typing import Callable, Tuple, Union

import gt4py
import gt4py as gt
import numpy
import xarray as xr
from fv3gfs.util import Quantity
from gt4py import gtscript

import fv3core
import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils

from .utils import global_config


ArgSpec = collections.namedtuple(
    "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
)
VALID_INTENTS = ["in", "out", "inout", "unknown"]


def state_inputs(*arg_specs):
    for spec in arg_specs:
        if spec.intent not in VALID_INTENTS:
            raise ValueError(
                f"intent for {spec.arg_name} is {spec.intent}, must be one of {VALID_INTENTS}"
            )

    def decorator(func):
        @functools.wraps(func)
        def wrapped(state, *args, **kwargs):
            namespace_kwargs = {}
            for spec in arg_specs:
                arg_name, standard_name, units, intent = spec
                if standard_name not in state:
                    raise ValueError(f"{standard_name} not present in state")
                elif units != state[standard_name].units:
                    raise ValueError(
                        f"{standard_name} has units {state[standard_name].units} when {units} is required"
                    )
                elif intent not in VALID_INTENTS:
                    raise ValueError(
                        f"expected intent to be one of {VALID_INTENTS}, got {intent}"
                    )
                else:
                    namespace_kwargs[arg_name] = state[standard_name].storage
                    namespace_kwargs[arg_name + "_quantity"] = state[standard_name]
            func(types.SimpleNamespace(**namespace_kwargs), *args, **kwargs)

        return wrapped

    return decorator


class FV3StencilObject:
    """GT4Py stencil object used for fv3core."""

    def __init__(self, stencil_object: gt4py.StencilObject, build_info: dict):
        self.stencil_object = stencil_object
        self._build_info = build_info

    @property
    def build_info(self) -> dict:
        """Return the build_info created when compiling the stencil."""
        return self._build_info

    def __call__(self, *args, **kwargs):
        return self.stencil_object(*args, **kwargs)


def _ensure_global_flags_not_specified_in_kwargs(stencil_kwargs):
    flag_errmsg = (
        "The {} flag should be set in "
        + __name__
        + " instead of as an argument to stencil"
    )
    for flag in ("rebuild", "backend"):
        if flag in stencil_kwargs:
            raise ValueError(flag_errmsg.format(flag))


def gtstencil(definition=None, **stencil_kwargs) -> Callable[..., None]:
    _ensure_global_flags_not_specified_in_kwargs(stencil_kwargs)

    def decorator(func) -> Callable[..., None]:
        stencils = {}

        @functools.wraps(func)
        def wrapped(*args, **kwargs) -> None:
            # This uses the module-level globals backend and rebuild (defined above)
            key = (global_config.get_backend(), global_config.get_rebuild())
            if key not in stencils:
                # Add globals to stencil_kwargs
                stencil_kwargs["rebuild"] = global_config.get_rebuild()
                stencil_kwargs["backend"] = global_config.get_backend()
                # Generate stencil
                build_info = {}
                stencil = gtscript.stencil(build_info=build_info, **stencil_kwargs)(
                    func
                )
                stencils[key] = FV3StencilObject(stencil, build_info)
            kwargs["splitters"] = kwargs.get(
                "splitters", spec.grid.splitters(origin=kwargs.get("origin"))
            )
            return stencils[key](*args, **kwargs)

        return wrapped

    if definition is None:
        return decorator
    else:
        return decorator(definition)
