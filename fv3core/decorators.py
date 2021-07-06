import collections
import collections.abc
import copy
import functools
import inspect
import os
import re
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
import dace.data
import gt4py
import gt4py.definitions
from gt4py import gtscript
from gt4py.storage.storage import Storage
from gt4py.utils import shash
import numpy as np

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


class FrozenStencil:
    """
    Wrapper for gt4py stencils which stores origin and domain at compile time,
    and uses their stored values at call time.

    This is useful when the stencil itself is meant to be used on a certain
    grid, for example if a compile-time external variable is tied to the
    values of origin and domain.
    """

    loaded_compiled_sdfgs: Dict[str, dace.SDFG] = dict()

    def __sdfg__(self, *args, **kwargs):
        if self._sdfg is not None:
            return copy.deepcopy(self._sdfg)

        if hasattr(self.stencil_object, "sdfg"):
            stencil_object = self.stencil_object
        else:
            stencil_kwargs = {**self.stencil_config.stencil_kwargs, "backend": "gtc:dace"}
            stencil_object = gtscript.stencil(definition=self.func, externals=self.externals, **stencil_kwargs)

        basename= os.path.splitext(stencil_object._file_name)[0]
        filename = basename +"_wrapper_" + str(shash(self.origin, self.domain))+".sdfg"
        try:
            self._sdfg = dace.SDFG.from_file(filename)
            print('reused (__sdfg__):', filename)
            return copy.deepcopy(self._sdfg)
        except FileNotFoundError:
            pass
        except Exception:
            raise
        inner_sdfg = stencil_object.sdfg
        self._sdfg = dace.SDFG('FrozenStencil_'+inner_sdfg.name)
        state = self._sdfg.add_state('FrozenStencil_'+inner_sdfg.name+"_state")

        inputs = set()
        outputs = set()
        for inner_state in inner_sdfg.nodes():
            for node in inner_state.nodes():
                if not isinstance(node, dace.nodes.AccessNode) or inner_sdfg.arrays[node.data].transient:
                    continue
                if node.access!=dace.dtypes.AccessType.WriteOnly:
                    inputs.add(node.data)
                if node.access != dace.dtypes.AccessType.ReadOnly:
                    outputs.add(node.data)


        nsdfg = state.add_nested_sdfg(inner_sdfg, None, inputs, outputs)
        for name, array in inner_sdfg.arrays.items():
            if isinstance(array, dace.data.Array) and not array.transient:
                axes = self.stencil_object.field_info[name].axes

                shape = [f"__{name}_{axis}_size" for axis in  axes] + [str(d) for d in self.stencil_object.field_info[name].data_dims]

                self._sdfg.add_array(name, dtype=array.dtype, strides=array.strides, shape=shape)
                if isinstance(self.origin, tuple):
                    origin = [o for a, o in zip("IJK", self.origin) if a in axes]
                else:
                    origin = self.origin.get(name, self.origin.get('_all_', None))
                    if len(origin)==3:
                        origin = [o for a, o in zip("IJK", origin) if a in axes]

                subset_strs = [f"{o-e}:{o-e+s}" for o, e ,s in zip(origin, self.stencil_object.field_info[name].boundary.lower_indices, inner_sdfg.arrays[name].shape)]
                subset_strs += [f"0:{d}" for d in self.stencil_object.field_info[name].data_dims]

                if name in inputs:
                    state.add_edge(state.add_read(name), None, nsdfg, name, dace.Memlet.simple(name, ",".join(subset_strs)))
                if name in outputs:
                    state.add_edge(nsdfg, name, state.add_write(name), None, dace.Memlet.simple(name, ",".join(subset_strs)))

        for symbol in nsdfg.sdfg.free_symbols:
            if symbol not in self._sdfg.symbols:
                self._sdfg.add_symbol(symbol, nsdfg.sdfg.symbols[symbol])
        for sdfg in self._sdfg.all_sdfgs_recursive():
            sdfg.replace("__I", str(self.domain[0]))
            sdfg.replace("__J", str(self.domain[1]))
            sdfg.replace("__K", str(self.domain[2]))
            sdfg.specialize({"__I": self.domain[0], "__J": self.domain[1], "__K": self.domain[2]})
        for _, name, array in self._sdfg.arrays_recursive():
            if array.transient:
                array.lifetime = dace.dtypes.AllocationLifetime.SDFG


        self._sdfg.arg_names = [arg for arg in self.func.__annotations__.keys() if arg != 'return']
        for arg in self._sdfg.arg_names:
            if arg in self.stencil_object.field_info and self.stencil_object.field_info[arg] is None:
                shape = tuple(dace.symbolic.symbol(f"__{arg}_{str(axis)}_size") for axis in self.func.__annotations__[arg].axes)
                strides = tuple(dace.symbolic.symbol(f"__{arg}_{str(axis)}_stride") for axis in self.func.__annotations__[arg].axes)
                self._sdfg.add_array(arg, shape=shape, strides=strides, dtype=dace.typeclass(str(self.func.__annotations__[arg].dtype)))
            if arg in self.stencil_object.parameter_info and self.stencil_object.parameter_info[arg] is None:
                self._sdfg.add_symbol(arg, stype=dace.typeclass(self.func.__annotations__[arg]))
        true_args = [arg for arg in self._sdfg.signature_arglist(with_types=False) if not re.match(f"__.*_._stride", arg) and not re.match(f"__.*_._size", arg)]
        assert len(self._sdfg.arg_names)==len(true_args)
        self._sdfg.save(filename)
        print('saved (__sdfg__):', filename)
        return dace.SDFG.from_json(self._sdfg.to_json())

    def __sdfg_closure__(self, *args, **kwargs):
        return {}

    def __sdfg_constant_args__(self):
        return []

    def __sdfg_argnames__(self):
        return [arg for arg in self.func.__annotations__.keys() if arg != 'return']

    def __init__(
        self,
        func: Callable[..., None],
        origin: Union[Index3D, Mapping[str, Tuple[int, ...]]],
        domain: Index3D,
        stencil_config: Optional[StencilConfig] = None,
        externals: Optional[Mapping[str, Any]] = None,
        jit=False
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

        self.domain = domain

        self.func = func

        self._sdfg = None

        if stencil_config is not None:
            self.stencil_config: StencilConfig = stencil_config
        else:
            self.stencil_config = global_config.get_stencil_config()

        if externals is None:
            externals = {}
        self.externals = externals

        self.stencil_object: gt4py.StencilObject = gtscript.stencil(
            definition=func,
            externals=externals,
            **self.stencil_config.stencil_kwargs,
        ) if not jit else None
        """generated stencil object returned from gt4py."""

        ref_stencil_kwargs = copy.deepcopy(self.stencil_config.stencil_kwargs)
        ref_stencil_kwargs['backend'] = "numpy"
        self.ref_stencil_object: gt4py.StencilObject = gtscript.stencil(
            definition=func,
            externals=externals,
            **ref_stencil_kwargs,)
        self._argument_names = tuple(inspect.getfullargspec(func).args)

        assert (
            len(self._argument_names) > 0
        ), "A stencil with no arguments? You may be double decorating"

        self._field_origins: Dict[str, Tuple[int, ...]] = compute_field_origins(
            self.stencil_object.field_info, self.origin
        ) if not jit else None
        """mapping from field names to field origins"""

        self._stencil_run_kwargs = {
            "_origin_": self._field_origins,
            "_domain_": self.domain,
        } if not jit else None

        self._written_fields = get_written_fields(self.stencil_object.field_info) if not jit else None

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> None:
        assert all( not hasattr(a, 'shape') or all(s > 0 for s in a.shape) for a in args)
        args_ref = copy.deepcopy(args)
        kwargs_ref = copy.deepcopy(kwargs)
        if self.stencil_config.validate_args:
            if __debug__ and "origin" in kwargs:
                raise TypeError("origin cannot be passed to FrozenStencil call")
            if __debug__ and "domain" in kwargs:
                raise TypeError("domain cannot be passed to FrozenStencil call")

            if self.stencil_object is None:
                self.stencil_object=gtscript.stencil(
                    definition=self.func,
                    externals=self.externals,
                    **{**self.stencil_config.stencil_kwargs, 'rebuild': True},
                )
                self._field_origins = compute_field_origins(
                    self.stencil_object.field_info, self.origin
                )
                self._stencil_run_kwargs = {
                    "_origin_": self._field_origins,
                    "_domain_": self.domain,
                }
                self._written_fields = get_written_fields(self.stencil_object.field_info)
            self.stencil_object(*args, **kwargs, origin=self._field_origins, domain=self.domain, validate_args=True)

        else:
            args_as_kwargs = dict(zip(self._argument_names, args))
            ref_args_as_kwargs = dict(zip(self._argument_names, args_ref))
            if False and self.stencil_config.backend=="gtc:dace":

                sym_dict = {name: (field.shape, field.strides) for name, field in {**args_as_kwargs, **kwargs}.items() if hasattr(self.func.__annotations__[name], 'axes')}

                try:
                    name = f"FrozenStencil_{self.func.__name__}_offset_wrapper{shash(self.origin, self.domain, sym_dict)}"
                    if name in self.loaded_compiled_sdfgs:
                        csdfg = self.loaded_compiled_sdfgs[name]
                    else:
                        csdfg = dace.sdfg.utils.load_precompiled_sdfg(dace.Config.get('default_build_folder')+os.sep+name)
                        self.loaded_compiled_sdfgs[name] = csdfg
                    print('reused (__call__):', dace.Config.get('default_build_folder')+os.sep+name)
                except (RuntimeError, FileNotFoundError):

                    basename = os.path.splitext(self.stencil_object._file_name)[0]
                    filename = basename + "_wrapper_" + str(shash(self.origin, self.domain, sym_dict)) + ".sdfg"
                    sdfg = self.__sdfg__()
                    sdfg.name += str(shash(self.origin, self.domain, sym_dict))
                    for name, (shape, strides) in sym_dict.items():
                        for i, axis in enumerate(self.func.__annotations__[name].axes):
                            sdfg.specialize({f"__{name}_{str(axis)}_size": shape[i]})
                            sdfg.specialize({f"__{name}_{str(axis)}_stride": strides[i] // self.func.__annotations__[name].dtype.itemsize})
                    csdfg = sdfg.compile()
                except Exception:
                    raise
                csdfg(**args_as_kwargs, **kwargs)
            else:
                if self.stencil_object is None:
                    self.stencil_object=gtscript.stencil(
                        definition=self.func,
                        externals=self.externals,
                        **{**self.stencil_config.stencil_kwargs, 'rebuild': self.func.__name__ == 'update_vorticity'},
                    )
                    self._field_origins = compute_field_origins(
                        self.stencil_object.field_info, self.origin
                    )
                    self._stencil_run_kwargs = {
                        "_origin_": self._field_origins,
                        "_domain_": self.domain,
                    }
                    self._written_fields = get_written_fields(self.stencil_object.field_info)
                self.stencil_object.run(**args_as_kwargs, **kwargs, **self._stencil_run_kwargs, exec_info=None)

            self.ref_stencil_object.run(
                **ref_args_as_kwargs, **kwargs_ref, **self._stencil_run_kwargs, exec_info=None
            )
            self._mark_cuda_fields_written({**args_as_kwargs, **kwargs})

            fail = False
            for name, arg in args_as_kwargs.items():
                ref = ref_args_as_kwargs[name]
                if hasattr(arg, 'data'):
                    try:
                        np.testing.assert_allclose(np.asarray(arg.data), np.asarray(ref.data),
                                                   err_msg=f'mismatch on arg {name}')
                    except AssertionError:
                        print(f'mismatch on arg {name}')
                        fail = True
            for k, v in kwargs.items():
                if hasattr(v, 'data'):
                    try:
                        np.testing.assert_allclose(np.asarray(v.data), np.asarray(kwargs_ref[k].data),
                                                   err_msg=f'mismatch on kwarg {k}')
                    except AssertionError:
                        print(f'mismatch on kwarg {k}')
                        fail = True
            assert not fail, str(self.func.__name__)
            # assert not self.func.__name__ == 'update_vorticity'


    def _mark_cuda_fields_written(self, fields: Mapping[str, Storage]):
        if "cuda" in self.stencil_config.backend:
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
        and field_info[field_name].access != gt4py.definitions.AccessKind.READ
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
