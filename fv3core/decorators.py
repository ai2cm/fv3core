import collections
import collections.abc
import functools
import inspect
import os.path
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

import dace
import gt4py
import gt4py.definitions
from dace.frontend.python.common import SDFGClosure, SDFGConvertible
from dace.frontend.python.parser import DaceProgram
from dace.transformation.auto.auto_optimize import make_transients_persistent
from dace.transformation.helpers import get_parent_map
from gt4py import gtscript
from gt4py.storage.storage import Storage

import fv3core
import fv3core._config as spec
import fv3core.utils
import fv3core.utils.global_config as global_config
import fv3core.utils.grid
from fv3core.utils.future_stencil import future_stencil
from fv3core.utils.global_config import StencilConfig
from fv3core.utils.mpi import MPI
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


class FrozenStencil(SDFGConvertible):
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
        skip_passes: Optional[Tuple[str, ...]] = None,
    ):
        """
        Args:
            func: stencil definition function
            origin: gt4py origin to use at call time
            domain: gt4py domain to use at call time
            stencil_config: container for stencil configuration
            externals: compile-time external variables required by stencil
            skip_passes: compiler passes to skip when building stencil
        """
        self.origin = origin
        self.domain = domain

        if stencil_config is not None:
            self.stencil_config: StencilConfig = stencil_config
        else:
            self.stencil_config = global_config.get_stencil_config()

        if externals is None:
            externals = {}

        stencil_function = gtscript.stencil
        stencil_kwargs = {**self.stencil_config.stencil_kwargs}

        # Enable distributed compilation if running in parallel
        if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
            stencil_function = future_stencil
            stencil_kwargs["wrapper"] = self

        if skip_passes and global_config.is_gtc_backend():
            stencil_kwargs["pass_order"] = {
                pass_name: None for pass_name in skip_passes
            }

        self.stencil_object: gt4py.StencilObject = stencil_function(
            definition=func,
            externals=externals,
            **stencil_kwargs,
        )
        """generated stencil object returned from gt4py."""

        self._argument_names = tuple(inspect.getfullargspec(func).args)

        assert (
            len(self._argument_names) > 0
        ), "A stencil with no arguments? You may be double decorating"

        field_info = self.stencil_object.field_info
        self._field_origins: Dict[str, Tuple[int, ...]] = compute_field_origins(
            field_info, self.origin
        )
        """mapping from field names to field origins"""

        self._stencil_run_kwargs: Dict[str, Any] = {
            "_origin_": self._field_origins,
            "_domain_": self.domain,
        }

        self._written_fields: List[str] = get_written_fields(field_info)

        self.sdfg_wrapper = gtscript.SDFGWrapper(
            definition=func,
            origin=origin,
            domain=domain,
            externals=externals,
            name=f"{__name__}.{func.__name__}",
            backend=self.stencil_config.backend,
        )

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> None:
        if any(d == 0 for d in self.domain):
            return
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
            self._mark_cuda_fields_written({**args_as_kwargs, **kwargs})

    def __sdfg__(self, *args, **kwargs):
        return self.sdfg_wrapper.__sdfg__(*args, **kwargs)

    def __sdfg_signature__(self):
        return self.sdfg_wrapper.__sdfg_signature__()

    def __sdfg_closure__(self, *args, **kwargs):
        return self.sdfg_wrapper.__sdfg_closure__(*args, **kwargs)

    def closure_resolver(self, constant_args, parent_closure=None):
        return SDFGClosure()

    def _mark_cuda_fields_written(self, fields: Mapping[str, Storage]):
        if global_config.is_gpu_backend():
            for write_field in self._written_fields:
                fields[write_field]._set_device_modified()


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


def get_written_fields(field_info) -> List[str]:
    """Returns the list of fields that are written.

    Args:
        field_info: field_info attribute of gt4py stencil object
    """
    write_fields = [
        field_name
        for field_name in field_info
        if field_info[field_name]
        and bool(field_info[field_name].access & gt4py.definitions.AccessKind.WRITE)
    ]
    return write_fields


def compute_field_origins(
    field_info_mapping, origin: Union[Index3D, Mapping[str, Tuple[int, ...]]]
) -> Dict[str, Tuple[int, ...]]:
    """
    Computes the origin for each field in the stencil call.

    Args:
        field_info_mapping: from stencil.field_info, a mapping which gives the
            dimensionality of each input field
        origin: the (i, j, k) coordinate of the origin

    Returns:
        origin_mapping: a mapping from field names to origins
    """
    if isinstance(origin, tuple):
        field_origins: Dict[str, Tuple[int, ...]] = {"_all_": origin}
        origin_tuple: Tuple[int, ...] = origin
    else:
        field_origins = {**origin}
        origin_tuple = origin["_all_"]
    field_names = tuple(field_info_mapping.keys())
    for i, field_name in enumerate(field_names):
        if field_name not in field_origins:
            field_info = field_info_mapping[field_name]
            if field_info is not None:
                field_origin_list = []
                for ax in field_info.axes:
                    origin_index = {"I": 0, "J": 1, "K": 2}[ax]
                    field_origin_list.append(origin_tuple[origin_index])
                field_origin = tuple(field_origin_list)
            else:
                field_origin = origin_tuple
            field_origins[field_name] = field_origin
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
            stencil_config = global_config.get_stencil_config()
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


class LazyComputepathFunction:
    def __init__(self, func, use_dace, skip_dacemode, load_sdfg):
        self.func = func
        self._use_dace = use_dace
        self._skip_dacemode = skip_dacemode
        self._load_sdfg = load_sdfg
        self.daceprog = dace.program(self.func)
        self._sdfg_loaded = False
        self._sdfg = None

    def __call__(self, *args, **kwargs):
        if self.use_dace:
            sdfg = self.__sdfg__(*args, **kwargs)
            return call_sdfg(
                self.daceprog,
                sdfg,
                args,
                kwargs,
                sdfg_final=(self._load_sdfg is not None),
            )
        else:
            return self.func(*args, **kwargs)

    @property
    def global_vars(self):
        return self.daceprog.global_vars

    @global_vars.setter
    def global_vars(self, value):
        self.daceprog.global_vars = value

    def __sdfg__(self, *args, **kwargs):
        if self._load_sdfg is None:
            return self.daceprog.to_sdfg(
                *args, **self.daceprog.__sdfg_closure__(), **kwargs, save=False
            )
        else:
            if not self._sdfg_loaded:
                if os.path.isfile(self._load_sdfg):
                    self.daceprog.load_sdfg(self._load_sdfg, *args, **kwargs)
                    self._sdfg_loaded = True
                else:
                    self.daceprog.load_precompiled_sdfg(
                        self._load_sdfg, *args, **kwargs
                    )
                    self._sdfg_loaded = True
            return next(iter(self.daceprog._cache.cache.values())).sdfg

    def __sdfg_closure__(self, *args, **kwargs):
        return self.daceprog.__sdfg_closure__(*args, **kwargs)

    def __sdfg_signature__(self):
        return self.daceprog.argnames, self.daceprog.constant_args

    def closure_resolver(self, constant_args, parent_closure=None):
        return self.daceprog.closure_resolver(constant_args, parent_closure)

    @property
    def use_dace(self):
        return self._use_dace or (
            global_config.get_dacemode() and not self._skip_dacemode
        )


class LazyComputepathMethod:

    bound_callables: Dict[Tuple[int, int], Callable] = dict()

    class SDFGEnabledCallable(SDFGConvertible):
        def __init__(self, lazy_method, obj_to_bind):
            methodwrapper = dace.method(lazy_method.func)
            self.obj_to_bind = obj_to_bind
            self.lazy_method = lazy_method
            self.daceprog = methodwrapper.__get__(obj_to_bind)

        @property
        def global_vars(self):
            return self.daceprog.global_vars

        @global_vars.setter
        def global_vars(self, value):
            self.daceprog.global_vars = value

        def __call__(self, *args, **kwargs):
            if self.lazy_method.use_dace:
                sdfg = self.__sdfg__(*args, **kwargs)
                return call_sdfg(
                    self.daceprog,
                    sdfg,
                    args,
                    kwargs,
                    sdfg_final=(self.lazy_method._load_sdfg is not None),
                )
            else:
                return self.lazy_method.func(self.obj_to_bind, *args, **kwargs)

        def __sdfg__(self, *args, **kwargs):
            if self.lazy_method._load_sdfg is None:
                return self.daceprog.to_sdfg(
                    *args, **self.daceprog.__sdfg_closure__(), **kwargs, save=False
                )
            else:
                if os.path.isfile(self.lazy_method._load_sdfg):
                    self.daceprog.load_sdfg(
                        self.lazy_method._load_sdfg, *args, **kwargs
                    )
                else:
                    self.daceprog.load_precompiled_sdfg(
                        self.lazy_method._load_sdfg, *args, **kwargs
                    )
                return self.daceprog.__sdfg__(*args, **kwargs)

        def __sdfg_closure__(self, reevaluate=None):
            return self.daceprog.__sdfg_closure__(reevaluate)

        def __sdfg_signature__(self):
            return self.daceprog.argnames, self.daceprog.constant_args

        def closure_resolver(self, constant_args, parent_closure=None):
            return self.daceprog.closure_resolver(constant_args, parent_closure)

    def __init__(self, func, use_dace, skip_dacemode, load_sdfg):
        self.func = func
        self._use_dace = use_dace
        self._skip_dacemode = skip_dacemode
        self._load_sdfg = load_sdfg

    def __get__(self, obj, objype=None):

        if (id(obj), id(self.func)) not in LazyComputepathMethod.bound_callables:

            LazyComputepathMethod.bound_callables[
                (id(obj), id(self.func))
            ] = LazyComputepathMethod.SDFGEnabledCallable(self, obj)

        return LazyComputepathMethod.bound_callables[(id(obj), id(self.func))]

    @property
    def use_dace(self):
        return self._use_dace or (
            global_config.get_dacemode() and not self._skip_dacemode
        )


def computepath_method(*args, **kwargs):
    skip_dacemode = kwargs.get("skip_dacemode", False)
    load_sdfg = kwargs.get("load_sdfg", None)
    use_dace = kwargs.get("use_dace", False)

    def _decorator(method):
        return LazyComputepathMethod(method, use_dace, skip_dacemode, load_sdfg)

    if len(args) == 1 and not kwargs and callable(args[0]):
        return _decorator(args[0])
    else:
        assert not args, "bad args"
        return _decorator


def computepath_function(
    *args, **kwargs
) -> Union[Callable[..., Any], LazyComputepathFunction]:
    skip_dacemode = kwargs.get("skip_dacemode", False)
    load_sdfg = kwargs.get("load_sdfg", None)
    use_dace = kwargs.get("use_dace", False)

    def _decorator(function):
        return LazyComputepathFunction(function, use_dace, skip_dacemode, load_sdfg)

    if len(args) == 1 and not kwargs and callable(args[0]):
        return _decorator(args[0])
    else:
        assert not args, "bad args"
        return _decorator
