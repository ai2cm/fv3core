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
                # if isinstance(state[standard_name].storage, numpy.ndarray):
                #     dat = state[standard_name].storage.copy()
                #     dims = state[standard_name].dims
                #     origin = namespace_kwargs[arg_name + "_quantity"].origin
                #     extent = namespace_kwargs[arg_name + "_quantity"].extent
                #     # if len(state[standard_name].storage.shape) == 2: #TODO maybe add handling for non (y,x) arrays
                #     #     dat = numpy.tile(dat, (1,1,spec.namelist.npz+1))
                #     #     dims = (dims[0], dims[1], "z")
                #     #     origin = (origin[0], origin[1], 0)
                #     #     extent = (extent[0], extent[1], spec.namelist.npz)
                #     # elif len(state[standard_name].storage.shape) == 1: #fml
                #     #     xdim = spec.namelist.npx+6# 3 halo on each side
                #     #     ydim = spec.namelist.npy+6# ditto
                #     #     zdim = spec.namelist.npz+1 #no halos in z-direction!
                #     #     dat = numpy.repeat(dat,xdim*ydim,axis=0).reshape(zdim,ydim,xdim)
                #     #     dims = ("x", "y", dims[0])#TODO maybe add handling for non z arrays
                #     #     origin = (3, 3, origin[0])
                #     #     extent = (spec.namelist.npx-1, spec.namelist.npy-1, extent[0])
                #     # elif len(state[standard_name].storage.shape)!=3:
                #     #     raise IndexError(
                #     #         f"{standard_name} is {len(state[standard_name].storage.shape)} dimensional, but arrays must have 1, 2, or 3 dimensions"
                #     # )
                #     # dat = dat.transpose(2,1,0)
                #     namespace_kwargs[arg_name] = gt.storage.from_array(data=dat, backend=fv3core.utils.gt4py_utils.backend, default_origin=(0,0,0), shape=(spec.namelist.npx+6,spec.namelist.npy+6,spec.namelist.npz+1))
                #     namespace_kwargs[arg_name + "_quantity"] = Quantity(data=namespace_kwargs[arg_name], units=namespace_kwargs[arg_name + "_quantity"].units, origin=origin, extent=extent, dims=dims)#?
            func(types.SimpleNamespace(**namespace_kwargs), *args, **kwargs)

        return wrapped

    return decorator
