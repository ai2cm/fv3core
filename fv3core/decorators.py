import collections
import functools
import hashlib
import os
import types
from typing import BinaryIO, Callable, Tuple, Union

import gt4py
import gt4py as gt
import numpy as np
import xarray as xr
import yaml
from fv3gfs.util import Quantity
from gt4py import gtscript

import fv3core
import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils

from .utils import global_config, mpi


ArgSpec = collections.namedtuple(
    "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
)
VALID_INTENTS = ["in", "out", "inout", "unknown"]
_STENCIL_LOGGER = None


def get_stencil_logger():
    global _STENCIL_LOGGER
    if _STENCIL_LOGGER is None:
        pass
    return _STENCIL_LOGGER


def enable_stencil_report(*, path: str, save_args: bool, save_report: bool):
    global stencil_report_path
    global save_stencil_args
    global save_stencil_report
    stencil_report_path = path
    save_stencil_args = save_args
    save_stencil_report = save_report
    print(
        "REPORT SETTINGS", stencil_report_path, save_stencil_args, save_stencil_report
    )


def disable_stencil_report():
    global stencil_report_path
    global save_stencil_args
    global save_stencil_report
    stencil_report_path = None
    save_stencil_args = False
    save_stencil_report = False


stencil_report_path = None
save_stencil_args = False
save_stencil_report = False


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


def gtstencil(definition=None, **stencil_kwargs) -> Callable[..., None]:
    if "rebuild" in stencil_kwargs:
        raise ValueError(
            f"The rebuild flag should be set in {__name__} instead of as an argument to stencil"
        )
    if "backend" in stencil_kwargs:
        raise ValueError(
            f"The backend flag should be set in {__name__} instead of as an argument to stencil"
        )

    def decorator(func) -> Callable[..., None]:
        stencils = {}
        times_called = 0

        @functools.wraps(func)
        def wrapped(*args, **kwargs) -> None:
            nonlocal times_called
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
                "splitters",
                spec.grid.splitters(origin=kwargs.get("origin")),
            )
            argnames = []
            name = func.__module__.split(".")[-1] + "." + func.__name__
            _maybe_save_report(name, times_called, func.__dict__['_gtscript_']['api_signature'], args, kwargs)
            times_called += 1
            return stencils[key](*args, **kwargs)

        return wrapped

    if definition is None:
        return decorator
    else:
        return decorator(definition)


def _get_case_name(name, times_called):
    return f"stencil-{name}-n{times_called:04d}"


def _get_report_filename():
    return f"stencil-report-r{spec.grid.rank:03d}.yml"


def _maybe_save_report(name, times_called, arg_infos, args, kwargs):
    case_name = _get_case_name(name, times_called)
    print(
        "REPORT SETTINGS", stencil_report_path, save_stencil_args, save_stencil_report
    )
    if save_stencil_args:
        args_filename = os.path.join(stencil_report_path, f"{case_name}.npz")
        with open(args_filename, "wb") as f:
            _save_args(f, args, kwargs)
    if save_stencil_report:
        report_filename = os.path.join(stencil_report_path, _get_report_filename())
        print(f"saving at {report_filename}")
        with open(report_filename, "a") as f:
            yaml.safe_dump({case_name: _get_stencil_report(arg_infos, args, kwargs)}, f)


def _save_args(file: BinaryIO, args, kwargs):
    args = list(args)
    kwargs_list = sorted(list(kwargs.items()))
    for i, arg in enumerate(args):
        if isinstance(arg, gt.storage.storage.Storage):
            args[i] = np.asarray(arg)
    for i, (name, value) in enumerate(kwargs_list):
        if isinstance(value, gt.storage.storage.Storage):
            kwargs_list[i] = (name, np.asarray(value))
    np.savez(file, *args, **dict(kwargs_list))


def _get_stencil_report(arg_infos, args, kwargs):
    return {
        "args": _get_args_report(arg_infos, args),
        "kwargs": _get_kwargs_report(kwargs),
    }


def _get_args_report(arg_infos, args):
    report = {}
    for argi in range(len(args)):
        report[arg_infos[argi].name] = _get_arg_report(args[argi])
    return report

def _get_kwargs_report(kwargs):
    return {name: _get_arg_report(value) for (name, value) in kwargs.items()}


def _get_arg_report(arg):
    if isinstance(arg, gt.storage.storage.Storage):
        arg = np.asarray(arg)
    if isinstance(arg, np.ndarray):
        islice = slice(spec.grid.is_, spec.grid.ie + 1)
        jslice = slice(spec.grid.js, spec.grid.je + 1)
        arg = arg[islice, jslice, :]
        return {
            "md5": hashlib.md5(arg.tobytes()).hexdigest(),
            "min": float(arg.min()),
            "max": float(arg.max()),
            "mean": float(arg.mean()),
            "std": float(arg.std()),
        }
    else:
        return str(arg)
