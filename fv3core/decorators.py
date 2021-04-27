import collections
import collections.abc
import functools
import inspect
import types
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple, cast

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
    def __init__(
        self,
        func: Callable[..., None],
        origin: Index3D,
        domain: Index3D,
        stencil_config: Optional[StencilConfig] = None,
        externals: Optional[Mapping[str, Any]] = None,
    ):
        self.origin: Tuple[int, ...] = origin

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
            self.stencil_object, self.origin
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
            print(self._field_origins, self.domain)
            self.stencil_object(
                *args,
                **kwargs,
                origin=self.origin,
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
    stencil, origin: Tuple[int, ...]
) -> Dict[str, Tuple[int, ...]]:
    """Computes the origin for each field in the stencil call."""
    field_origins: Dict[str, Tuple[int, ...]] = {"_all_": origin}
    field_names = tuple(stencil.field_info.keys())
    for i in range(len(field_names)):
        field_name = field_names[i]
        field_info = stencil.field_info[field_name]
        if field_info is not None:
            field_origin_list = []
            for ax in field_info.axes:
                field_origin_list.append(origin[{"I": 0, "J": 1, "K": 2}[ax]])
            field_origin = tuple(field_origin_list)
        else:
            field_origin = origin
        field_origins[field_name] = field_origin
    return field_origins


def gtstencil(
    func,
    origin: Optional[Index3D] = None,
    domain: Optional[Index3D] = None,
    stencil_config: Optional[StencilConfig] = None,
    externals=None,
):
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


def get_non_frozen_stencil(func, externals) -> Callable[[Any], FrozenStencil]:
    stencil_dict: Dict[Tuple[Index3D, Index3D], FrozenStencil] = {}
    return _get_decorated(func, stencil_dict, externals)


def _get_decorated(
    func,
    stencil_dict: MutableMapping[Tuple[Index3D, Index3D], FrozenStencil],
    externals,
):
    # separated this code into its own routine for easier unit testing

    @functools.wraps(func)
    def decorated(*args, origin=None, domain=None, **kwargs):
        stencil_config = global_config.get_stencil_config()
        key = (origin, domain, stencil_config)
        if key not in stencil_dict:
            axis_offsets = fv3core.utils.grid.axis_offsets(
                spec.grid, origin=origin, domain=domain
            )
            stencil_dict[key] = FrozenStencil(
                func,
                origin,
                domain,
                stencil_config=stencil_config,
                externals={**axis_offsets, **externals},
            )
        return stencil_dict[key](*args, **kwargs)

    return decorated
