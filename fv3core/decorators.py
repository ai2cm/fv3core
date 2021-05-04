import collections
import collections.abc
import functools
import inspect
import types
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Tuple, Union, cast

import gt4py
from gt4py import gtscript

import fv3core
import fv3core._config as spec
import fv3core.utils
import fv3core.utils.global_config as global_config
import fv3core.utils.grid
from fv3core.utils.global_config import StencilConfig
from fv3core.utils.typing import Index3D


ArgSpec = collections.namedtuple(
    "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
)
VALID_INTENTS = ["in", "out", "inout", "unknown"]


def state_inputs(*arg_specs):
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


def ensure_reconstructed_args_match_defined(stencil_object, reconstructed_args):
    spec = inspect.getfullargspec(stencil_object.definition_func)
    definition_args = tuple(spec.args) + tuple(spec.kwonlyargs)
    if not definition_args == reconstructed_args:
        raise TypeError(
            "Argument list constructed from field and parameter info "
            "does not match stencil signature, possibly because "
            "parameters (e.g. float) must come after fields "
            f"in stencil arguments. Defined is {definition_args}, "
            f"reconstructed is {reconstructed_args}."
        )


class FrozenStencil:
    """
    Wrapper for gt4py stencils which stores origin and domain at compile time,
    and uses their stored values at call time.

    This is useful when the stencil itself is meant to be used on a certain
    grid, for example if a compile-time external variable is tied to the
    values of origin and domain.
    """

    def __init__(
        self,
        func: Callable[..., None],
        origin: Union[Index3D, Mapping[str, Tuple[int, ...]]],
        domain: Index3D,
        stencil_config: Optional[StencilConfig] = None,
        externals: Optional[Mapping[str, Any]] = None,
    ):
        """
        Args:
            func: stencil definition function
            origin: gt4py origin to use at call time
            domain: gt4py domain to use at call time
            stencil_config: container for stencil configuration
            externals: compile-time external variables required by stencil
        """
        self.origin = origin

        self.domain: Optional[Index3D] = domain

        if stencil_config is not None:
            self.stencil_config: StencilConfig = stencil_config
        else:
            self.stencil_config = global_config.get_stencil_config()

        if externals is None:
            externals = {}

        self.stencil_object: gt4py.StencilObject = gtscript.stencil(
            definition=func,
            externals=externals,
            **self.stencil_config.stencil_kwargs,
        )
        """generated stencil object returned from gt4py."""
        self._argument_names = self.field_names + self.parameter_names
        ensure_reconstructed_args_match_defined(
            self.stencil_object, self._argument_names
        )

        self._field_origins: Dict[str, Tuple[int, ...]] = compute_field_origins(
            self.stencil_object.field_info, self.origin
        )
        """mapping from field names to field origins"""

        self._stencil_run_kwargs = {
            "_origin_": self._field_origins,
            "_domain_": self.domain,
        }

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> None:
        if self.stencil_config.validate_args:
            if __debug__ and "origin" in kwargs:
                raise TypeError("origin cannot be passed to FrozenStencil call")
            if __debug__ and "domain" in kwargs:
                raise TypeError("domain cannot be passed to FrozenStencil call")
            self.stencil_object(
                *args,
                **kwargs,
                origin=self._field_origins,
                domain=self.domain,
                validate_args=True,
            )
        else:
            args_as_kwargs = dict(zip(self._argument_names, args))
            self.stencil_object.run(
                **args_as_kwargs, **kwargs, **self._stencil_run_kwargs, exec_info=None
            )

    @property
    def field_names(self) -> Tuple[str]:
        """names of stencil field call arguments"""
        return cast(Tuple[str], tuple(self.stencil_object.field_info.keys()))

    @property
    def parameter_names(self) -> Tuple[str]:
        """names of stencil parameter call arguments"""
        return cast(Tuple[str], tuple(self.stencil_object.parameter_info.keys()))


def compute_field_origins(
    field_info_mapping, origin: Union[Index3D, Mapping[str, Tuple[int, ...]]]
) -> Dict[str, Tuple[int, ...]]:
    """Computes the origin for each field in the stencil call."""
    if isinstance(origin, tuple):
        field_origins: Dict[str, Tuple[int, ...]] = {"_all_": origin}
        origin_tuple: Tuple[int, ...] = origin
    else:
        field_origins = {**origin}
        origin_tuple = origin["_all_"]
    for name, field_info in field_info_mapping:
        if field_info is not None:
            field_origin = field_origins.get(name, None)
            if field_origin is not None:
                field_origin_ndim = len(field_origin)
                if field_origin_ndim != field_info.ndim:
                    assert field_origin_ndim == field_info.domain_ndim, (
                        f"Invalid origin specification ({field_origin})"
                        f" for '{name}' field."
                    )
                    field_origins[name] = (
                        *field_origin,
                        *((0,) * len(field_info.data_dims)),
                    )
            elif origin_tuple is not None:
                field_origins[name] = (
                    *gt4py.utils.filter_mask(origin_tuple, field_info.domain_mask),
                    *((0,) * len(field_info.data_dims)),
                )
            else:
                raise ValueError(f"Missing origin for {name}")
    return field_origins


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
            configuration at the time of calling this function is used
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
            stencil_config = global_config.get_stencil_config()
        stencil = FrozenStencil(
            func,
            origin=origin,
            domain=domain,
            stencil_config=stencil_config,
            externals=externals,
        )
    return stencil


def get_non_frozen_stencil(func, externals) -> Callable[..., None]:
    stencil_dict: Dict[Hashable, FrozenStencil] = {}

    @functools.wraps(func)
    def decorated(
        *args,
        origin: Union[Index3D, Mapping[str, Tuple[int, ...]]],
        domain: Index3D,
        **kwargs,
    ):
        stencil_config = global_config.get_stencil_config()
        if isinstance(origin, Mapping):
            origin_key: Hashable = tuple(sorted(origin.items(), key=lambda x: x[0]))
            origin_tuple: Tuple[int, ...] = origin["_all_"]
        else:
            origin_key = origin
            origin_tuple = origin
        key: Hashable = (origin_key, domain, stencil_config)
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
