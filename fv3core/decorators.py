import collections
import collections.abc
import functools
import types
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
)

import dace
from dace.frontend.python.parser import DaceProgram
from dace.transformation.auto.auto_optimize import make_transients_persistent
from dace.transformation.helpers import get_parent_map

import gt4py
import gt4py.definitions
import types
from typing import Any, Callable, List, Mapping, Optional

import fv3core
import fv3core._config as spec
import fv3core.utils
import fv3core.utils.grid
from fv3core.utils import global_config
from fv3core.utils.stencil import FrozenStencil, StencilFactory
from fv3core.utils.typing import Index3D


ArgSpec = collections.namedtuple(
    "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
)
VALID_INTENTS = ["in", "out", "inout", "unknown"]


def to_gpu(sdfg: dace.SDFG):
    allmaps = [
        (me, state)
        for me, state in sdfg.all_nodes_recursive()
        if isinstance(me, dace.nodes.MapEntry)
    ]
    topmaps = [
        (me, state) for me, state in allmaps if get_parent_map(state, me) is None
    ]

    for sd, aname, arr in sdfg.arrays_recursive():
        if arr.shape == (1,):
            arr.storage = dace.StorageType.Register
        else:
            arr.storage = dace.StorageType.GPU_Global

    for mapentry, state in topmaps:
        mapentry.schedule = dace.ScheduleType.GPU_Device

    for sd in sdfg.all_sdfgs_recursive():
        sd.openmp_sections = False


def call_sdfg(daceprog: DaceProgram, sdfg: dace.SDFG, args, kwargs, sdfg_final=False):
    if not sdfg_final:
        if global_config.is_gpu_backend():
            to_gpu(sdfg)
            make_transients_persistent(sdfg=sdfg, device=dace.dtypes.DeviceType.GPU)
        else:
            make_transients_persistent(sdfg=sdfg, device=dace.dtypes.DeviceType.CPU)
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, gt4py.storage.Storage):
            arg.host_to_device()

    if not sdfg_final:
        sdfg_kwargs = daceprog._create_sdfg_args(sdfg, args, kwargs)
        for k in daceprog.constant_args:
            if k in sdfg_kwargs:
                del sdfg_kwargs[k]
        sdfg_kwargs = {k: v for k, v in sdfg_kwargs.items() if v is not None}
        res = sdfg(**sdfg_kwargs)
    else:
        res = daceprog(*args, **kwargs)
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, gt4py.storage.Storage) and hasattr(
            arg, "_set_device_modified"
        ):
            arg._set_device_modified()
    if res is not None:
        if global_config.is_gpu_backend():
            res = [
                gt4py.storage.from_array(
                    r,
                    default_origin=(0, 0, 0),
                    backend=global_config.get_backend(),
                    managed_memory=True,
                )
                for r in res
            ]
        else:
            res = [
                gt4py.storage.from_array(
                    r, default_origin=(0, 0, 0), backend=global_config.get_backend()
                )
                for r in res
            ]
    return res


def state_inputs(*arg_specs: ArgSpec):
    for sp in arg_specs:
        if sp.intent not in VALID_INTENTS:
            raise ValueError(
                f"intent for {sp.arg_name} is {sp.intent}, "
                "must be one of {VALID_INTENTS}"
            )

    def decorator(func):
        @functools.wraps(func)
        def wrapped(state, *args, **kwargs):
            namespace = get_namespace(arg_specs, state)
            func(namespace, *args, **kwargs)

        return wrapped

    return decorator


def get_namespace(arg_specs, state):
    namespace_kwargs = {}
    for sp in arg_specs:
        arg_name, standard_name, units, intent = sp
        if standard_name not in state:
            raise ValueError(f"{standard_name} not present in state")
        elif units != state[standard_name].units:
            raise ValueError(
                f"{standard_name} has units "
                f"{state[standard_name].units} when {units} is required"
            )
        elif intent not in VALID_INTENTS:
            raise ValueError(
                f"expected intent to be one of {VALID_INTENTS}, got {intent}"
            )
        else:
            namespace_kwargs[arg_name] = state[standard_name].storage
            namespace_kwargs[arg_name + "_quantity"] = state[standard_name]
    return types.SimpleNamespace(**namespace_kwargs)


def get_stencils_with_varied_bounds(
    func: Callable[..., None],
    origins: List[Index3D],
    domains: List[Index3D],
    stencil_factory: StencilFactory,
    externals: Optional[Mapping[str, Any]] = None,
) -> List[FrozenStencil]:
    assert len(origins) == len(domains), (
        "Lists of origins and domains need to have the same length, you provided "
        + str(len(origins))
        + " origins and "
        + str(len(domains))
        + " domains"
    )
    if externals is None:
        externals = {}
    stencils = []
    for origin, domain in zip(origins, domains):
        ax_offsets = fv3core.utils.grid.axis_offsets(spec.grid, origin, domain)
        stencils.append(
            stencil_factory.from_origin_domain(
                func,
                origin=origin,
                domain=domain,
                externals={**externals, **ax_offsets},
            )
        )
    return stencils
