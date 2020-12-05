import collections
import functools
import hashlib
import inspect
import os
import types
from typing import BinaryIO, Callable, Dict, Optional, Sequence, Tuple, Union

import gt4py
import gt4py.ir as gt_ir
import gt4py.storage as gt_storage
import numpy as np
import xarray as xr
import yaml
from fv3gfs.util import Quantity
from gt4py import gtscript

import fv3core
import fv3core._config as spec
import fv3core.utils
import fv3core.utils.gt4py_utils as utils
from fv3core.utils.typing import Int3

from .utils import global_config


ArgSpec = collections.namedtuple(
    "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
)
VALID_INTENTS = ["in", "out", "inout", "unknown"]


def enable_stencil_report(
    *, path: str, save_args: bool, save_report: bool, include_halos: bool = False
):
    global stencil_report_path
    global save_stencil_args
    global save_stencil_report
    global report_include_halos
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    stencil_report_path = path
    save_stencil_args = save_args
    save_stencil_report = save_report
    report_include_halos = include_halos


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
report_include_halos = False


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


def _ensure_global_flags_not_specified_in_kwargs(stencil_kwargs):
    flag_errmsg = "The {} flag should be set in fv3core.utils.global_config.py instead of as an argument to stencil"
    for flag in ("rebuild", "backend"):
        if flag in stencil_kwargs:
            raise ValueError(flag_errmsg.format(flag))


class FV3StencilObject:
    """GT4Py stencil object used for fv3core."""

    def __init__(self, func: Callable[..., None], **kwargs):
        self.func: Callable[..., None] = func
        """The definition function."""

        self.stencil_object: Optional[gt4py.StencilObject] = None
        """The generated stencil object returned from gt4py."""

        self.build_info: Optional[Dict[str, Any]] = None
        """Return the build_info created when compiling the stencil."""

        self.times_called: int = 0
        """Number of times this stencil has been called."""

        self.externals: Dict[str, Any] = kwargs.pop("externals", {})
        """Externals dictionary used for stencil generation."""

        self.backend_kwargs: Dict[str, Any] = kwargs
        """Remainder of the arguments are assumed to be gt4py compiler backend options."""

    @property
    def built(self) -> bool:
        """Returns whether the stencil is already built (if it has been called at least once)."""
        return self.stencil_object is not None

    @property
    def definition(self) -> Optional[gt_ir.StencilDefinition]:
        """Current stencil definition IR if built, else None."""
        return self.build_info["def_ir"]

    @property
    def implementation(self) -> Optional[gt_ir.StencilImplementation]:
        """Current stencil implementation IR if built, else None."""
        return self.build_info["impl_ir"]

    @property
    def externals(self) -> Dict[str, Any]:
        """Return a dictionary of external values used in the stencil generation."""
        externals_or_none = self.definition.externals if self.built else {}
        return externals_or_none or {}

    def __call__(self, *args, origin: Int3, domain: Int3, **kwargs):
        """Call the stencil, compiling the stencil if necessary.

        The stencil needs to be recompiled if any of the following changes
        1. the origin and/or domain
        2. any external value
        3. the function signature or code

        Args:
            domain: Stencil compute domain (required)
            origin: Data index mapped to (0, 0, 0) in the compute domain (required)
            externals: Dictionary of externals for the stencil call
        """

        stencil_kwargs = {
            "rebuild": global_config.get_rebuild(),
            "backend": global_config.get_backend(),
            "externals": {
                "namelist": spec.namelist,
                "grid": spec.grid,
                **fv3core.utils.axis_offsets(spec.grid, origin, domain),
                **self.externals,
            },
            **self.backend_kwargs,
        }

        regenerate_stencil = any(
            stencil_kwargs["externals"][key] != value for key, value in self.externals
        )

        if regenerate_stencil or stencil_kwargs["rebuild"]:
            new_build_info = {}
            stencil_object = gtscript.stencil(
                definition=self.func, build_info=new_build_info, **stencil_kwargs
            )
            # If the hash changes, there is updated build_info
            if hash(self.stencil_object) != hash(stencil_object):
                self.build_info = new_build_info
            # The stencil object always changes
            self.stencil_object = stencil_object

        # Call it
        kwargs["validate_args"] = kwargs.get("validate_args", utils.validate_args)
        name = f"{self.func.__module__}.{self.func.__name__}"
        _maybe_save_report(
            f"{name}-before",
            self.times_called,
            self.func.__dict__["_gtscript_"]["api_signature"],
            args,
            kwargs,
        )
        self.stencil_object(*args, **kwargs, origin=origin, domain=domain)
        _maybe_save_report(
            f"{name}-after",
            self.times_called,
            self.func.__dict__["_gtscript_"]["api_signature"],
            args,
            kwargs,
        )
        self.times_called += 1


def gtstencil(definition=None, **stencil_kwargs) -> Callable[..., None]:
    _ensure_global_flags_not_specified_in_kwargs(stencil_kwargs)

    def decorator(func) -> FV3StencilObject:
        return FV3StencilObject(func, **stencil_kwargs)

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
    if save_stencil_args:
        args_filename = os.path.join(stencil_report_path, f"{case_name}.npz")
        with open(args_filename, "wb") as f:
            _save_args(f, args, kwargs)
    if save_stencil_report:
        report_filename = os.path.join(stencil_report_path, _get_report_filename())
        with open(report_filename, "a") as f:
            yaml.safe_dump({case_name: _get_stencil_report(arg_infos, args, kwargs)}, f)


def _save_args(file: BinaryIO, args, kwargs):
    args = list(args)
    kwargs_list = sorted(list(kwargs.items()))
    for i, arg in enumerate(args):
        if isinstance(arg, gt_storage.storage.Storage):
            args[i] = np.asarray(arg)
    for i, (name, value) in enumerate(kwargs_list):
        if isinstance(value, gt_storage.storage.Storage):
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
    if isinstance(arg, gt_storage.storage.Storage):
        arg = np.asarray(arg)
    if isinstance(arg, np.ndarray):
        if not report_include_halos:
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
