import collections
import collections.abc
import functools
import types
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

import fv3core
import fv3core._config as spec
import fv3core.utils
import fv3core.utils.global_config as global_config
import fv3core.utils.grid
from fv3core.utils.stencil import FrozenStencil, StencilConfig
from fv3core.utils.typing import Index3D


ArgSpec = collections.namedtuple(
    "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
)
VALID_INTENTS = ["in", "out", "inout", "unknown"]


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
    stencil_config: Optional[StencilConfig] = None,
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
            FrozenStencil(
                func,
                origin=origin,
                domain=domain,
                stencil_config=stencil_config,
                externals={**externals, **ax_offsets},
            )
        )
    return stencils


def gtstencil(
    func,
    origin: Optional[Index3D] = None,
    domain: Optional[Index3D] = None,
    stencil_config: Optional[StencilConfig] = None,
    externals: Optional[Mapping[str, Any]] = None,
):
    """
    Returns a wrapper over gt4py stencils.

    If origin and domain are not given, they must be provided at call time,
    and a separate stencil is compiled for each origin and domain pair used.

    Args:
        func: stencil definition function
        origin: the start of the compute domain
        domain: the size of the compute domain, required if origin is given
        stencil_config: stencil configuration, by default global stencil
            configuration at the first call time is used
        externals: compile-time constants used by stencil

    Returns:
        wrapped_stencil: an object similar to gt4py stencils, takes origin
            and domain as arguments if and only if they were not given
            as arguments to gtstencil
    """
    if not (origin is None) == (domain is None):
        raise TypeError("must give both origin and domain arguments, or neither")
    if externals is None:
        externals = {}
    if origin is None:
        stencil = get_non_frozen_stencil(func, externals)
    else:
        # TODO: delete this global default
        if stencil_config is None:
            stencil_config = get_global_stencil_config()
        stencil = FrozenStencil(
            func,
            origin=origin,
            domain=domain,
            stencil_config=stencil_config,
            externals=externals,
        )
    return stencil


def get_global_stencil_config() -> StencilConfig:
    return StencilConfig(
        backend=global_config.get_backend(),
        rebuild=global_config.get_rebuild(),
        validate_args=global_config.get_validate_args(),
    )


def get_non_frozen_stencil(func, externals) -> Callable[..., None]:
    stencil_dict: Dict[Hashable, FrozenStencil] = {}
    # must use a mutable container object here to hold config,
    # `global` does not work in this case. Cannot retreve StencilConfig
    # yet because it is not set at import time, when this function
    # is called by gtstencil throughout the repo
    # so instead we retrieve it at first call time
    stencil_config_holder: List[StencilConfig] = []

    @functools.wraps(func)
    def decorated(
        *args,
        origin: Union[Index3D, Mapping[str, Tuple[int, ...]]],
        domain: Index3D,
        **kwargs,
    ):
        try:
            stencil_config = stencil_config_holder[0]
        except IndexError:
            stencil_config = get_global_stencil_config()
            stencil_config_holder.append(stencil_config)
        try:  # works if origin is a Mapping
            origin_key: Hashable = tuple(
                sorted(origin.items(), key=lambda x: x[0])  # type: ignore
            )
            origin_tuple: Tuple[int, ...] = origin["_all_"]  # type: ignore
        except AttributeError:  # assume origin is a tuple
            origin_key = origin
            origin_tuple = cast(Index3D, origin)
        # rank is needed in the key for regression testing
        # for > 6 ranks, where each rank may or may not be
        # on a tile edge
        key: Hashable = (origin_key, domain, spec.grid.rank)
        if key not in stencil_dict:
            axis_offsets = fv3core.utils.grid.axis_offsets(
                spec.grid, origin=origin_tuple, domain=domain
            )
            stencil_dict[key] = FrozenStencil(
                func,
                origin,
                domain,
                stencil_config=stencil_config,
                externals={**axis_offsets, **externals, "namelist": spec.namelist},
            )
        return stencil_dict[key](*args, **kwargs)

    return decorated
