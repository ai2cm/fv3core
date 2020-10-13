#!/usr/bin/env python3

import copy
import functools
import logging
import math
from typing import Any, Callable, Tuple, Union

import gt4py as gt
import gt4py.ir as gt_ir
import numpy as np
from gt4py import gtscript

# Problem: creates circular dependency
from fv3core.utils.mpi import MPI

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger("fv3ser")
backend = None  # Options: numpy, gtmc, gtx86, gtcuda, debug, dawn:gtmc
rebuild = True
managed_memory = True
_dtype = np.float_
sd = gtscript.Field[_dtype]
si = gtscript.Field[np.int_]
halo = 3
origin = (halo, halo, 0)
# TODO get from field_table
tracer_variables = [
    "qvapor",
    "qliquid",
    "qrain",
    "qice",
    "qsnow",
    "qgraupel",
    "qo3mr",
    "qsgs_tke",
    "qcld",
]

# Union of valid data types (from gt4py.gtscript)
DTypes = Union[bool, np.bool, int, np.int32, np.int64, float, np.float32, np.float64]


# 1 indexing to 0 and halos: -2, -1, 0 --> 0, 1,2
if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
    gt.config.cache_settings["dir_name"] = ".gt_cache_{:0>6d}".format(
        MPI.COMM_WORLD.Get_rank()
    )


# TODO remove when using quantities throughout model
def quantity_name(name):
    return name + "_quantity"


def make_storage(
    data_or_shape: Any,
    shape: Tuple[int, int, int] = None,
    origin: Tuple[int, int, int] = origin,
    dtype: DTypes = np.float64,
    init: bool = True,
    mask: Tuple[bool, bool, bool] = None,
    start: Tuple[int, int, int] = (0, 0, 0),
    dummy: Tuple[int, int, int] = None,
    names: Tuple[str, str, str] = None,
    axis: int = 2,
):
    """Create a new gt4py storage of a given shape.

    Args:
        shape: Size of the new storage
        origin: Default origin for gt4py stencil calls
        dtype: Data type
        init: If True, initializes the storage to the default value for the type

    Returns:
        gtscript.Field[dtype]: New storage
    """
    if isinstance(data_or_shape, tuple):
        shape = data_or_shape
        storage = gt.storage.empty(
            backend=backend,
            default_origin=origin,
            shape=shape,
            dtype=dtype,
            mask=mask,
            managed_memory=managed_memory,
        )
        if init:
            storage[:] = dtype()
    else:
        data = data_or_shape
        if shape is None:
            shape = data.shape

        # make_storage_2d:
        if len(data.shape) == 2:
            # axis refers to which axis should be repeated (when making a full 3d data),
            # dummy refers to a singleton axis
            if dummy or axis != 2:
                d_axis = dummy[0] if dummy else axis
                shape2d = shape[:d_axis] + shape[d_axis + 1 :]
            else:
                shape2d = shape[0:2]
            istart, jstart = start[0:2]
            isize, jsize = data.shape
            full_np_arr_2d = np.zeros(shape2d)
            full_np_arr_2d[istart : istart + isize, jstart : jstart + jsize] = asarray(
                data, type(full_np_arr_2d)
            )
            if dummy:
                full_np_arr_3d = full_np_arr_2d.reshape(shape)
            else:
                full_np_arr_3d = np.repeat(
                    full_np_arr_2d[:, :, np.newaxis], shape[axis], axis=2
                )
                if axis != 2:
                    full_np_arr_3d = np.moveaxis(full_np_arr_3d, 2, axis)
            data = full_np_arr_3d
        # make_storage_1d:
        elif len(data.shape) == 1:
            # axis refers to a repeated axis, dummy refers to a singleton axis
            kstart = start[2]
            if dummy:
                axis = list(set((0, 1, 2)).difference(dummy))[0]
            tilespec = list(shape)
            full_1d = np.zeros(shape[axis])
            full_1d[kstart : kstart + len(data)] = data
            tilespec[axis] = 1
            if dummy:
                if len(dummy) == len(tilespec) - 1:
                    r = full_1d.reshape((shape))
                else:
                    # TODO maybe, this is a little silly (repeat the array, then squash the dim), though eventually
                    # we shouldn't need this general capability if we refactor stencils to operate on 3d
                    full_1d = make_storage(
                        data, shape, start=(0, 0, kstart), origin=origin, axis=axis
                    )
                    dimslice = [slice(None)] * len(tilespec)
                    for dummy_axis in dummy:
                        dimslice[dummy_axis] = slice(0, 1)
                    r = full_1d[tuple(dimslice)]
            else:
                if axis == 2:
                    r = np.tile(full_1d, tuple(tilespec))
                elif axis == 1:
                    x = np.repeat(full_1d[np.newaxis, :], shape[0], axis=0)
                    r = np.repeat(x[:, :, np.newaxis], shape[2], axis=2)
                else:
                    y = np.repeat(full_1d[:, np.newaxis], shape[1], axis=1)
                    r = np.repeat(y[:, :, np.newaxis], shape[2], axis=2)
            data = r
        # make_storage_4d:
        elif len(data.shape) == 4:
            if names is None:
                raise Exception("for 4d variable storages, specify a list of names")
            data_dict = {}
            size4d = data.shape[3]
            for i in range(size4d):
                data_dict[names[i]] = make_storage(
                    np.squeeze(data[:, :, :, i]),
                    shape,
                    origin=origin,
                    start=start,
                    dummy=dummy,
                    axis=axis,
                )
            return data_dict
        # make_storage_3d:
        else:
            full_np_arr = np.zeros(shape)
            istart, jstart, kstart = start
            isize, jsize, ksize = data.shape
            full_np_arr[
                istart : istart + isize,
                jstart : jstart + jsize,
                kstart : kstart + ksize,
            ] = asarray(data, type(full_np_arr))
            data = full_np_arr

        storage = gt.storage.from_array(
            data=data,
            backend=backend,
            default_origin=origin,
            shape=shape,
            dtype=dtype,
            mask=mask,
            managed_memory=managed_memory,
        )

    return storage


def storage_dict(st_dict, names, shape, origin):
    for name in names:
        st_dict[name] = make_storage(shape, origin)


def k_slice_operation(key, value, ki, dictionary):
    if isinstance(value, gt.storage.storage.Storage):
        dictionary[key] = make_storage(
            value[:, :, ki], (value.shape[0], value.shape[1], len(ki))
        )
    else:
        dictionary[key] = value


def k_slice_inplace(data_dict, ki):
    for k, v in data_dict.items():
        k_slice_operation(k, v, ki, data_dict)


def k_slice(data_dict, ki):
    new_dict = {}
    for k, v in data_dict.items():
        k_slice_operation(k, v, ki, new_dict)
    return new_dict


def k_subset_run(func, data, splitvars, ki, outputs, grid_data, grid, allz=False):
    grid.npz = len(ki)
    grid.slice_data_k(ki)
    d = k_slice(data, ki)
    d.update(splitvars)
    results = func(**d)
    collect_results(d, results, outputs, ki, allz)
    grid.add_data(grid_data)


def collect_results(data, results, outputs, ki, allz=False):
    outnames = list(outputs.keys())
    endz = None if allz else -1
    logger.debug("Computing results for k indices: {}".format(ki[:-1]))
    for k in outnames:
        if k in data:
            # passing fields with single item in 3rd dimension leads to errors
            outputs[k][:, :, ki[:endz]] = data[k][:, :, :endz]
    if results is not None:
        for ri in range(len(results)):
            outputs[outnames[ri]][:, :, ki[:endz]] = results[ri][:, :, :endz]


def k_split_run_dataslice(
    func, data, k_indices_array, splitvars_values, outputs, grid, allz=False
):
    num_k = grid.npz
    grid_data = copy.deepcopy(grid.data_fields)
    for ki in k_indices_array:
        splitvars = {}
        for name, value_array in splitvars_values.items():
            splitvars[name] = value_array[ki[0]]
        k_subset_run(func, data, splitvars, ki, outputs, grid_data, grid, allz)
    grid.npz = num_k


def get_kstarts(column_info, npz):
    compare = None
    kstarts = []
    for k in range(npz):
        column_vals = {}
        for q, v in column_info.items():
            if k < len(v):
                column_vals[q] = v[k]
        if column_vals != compare:
            kstarts.append(k)
            compare = column_vals
    for i in range(len(kstarts) - 1):
        kstarts[i] = (kstarts[i], kstarts[i + 1] - kstarts[i])
    kstarts[-1] = (kstarts[-1], npz - kstarts[-1])
    return kstarts


def k_split_run(func, data, k_indices, splitvars_values):
    for ki, nk in k_indices:
        splitvars = {}
        for name, value_array in splitvars_values.items():
            splitvars[name] = value_array[ki]
        data.update(splitvars)
        data["kstart"] = ki
        data["nk"] = nk
        logger.debug(
            "Running kstart: {}, num k:{}, variables:{}".format(ki, nk, splitvars)
        )
        func(**data)


def kslice_from_inputs(kstart, nk, grid):
    if nk is None:
        nk = grid.npz - kstart
    kslice = slice(kstart, kstart + nk)
    return [kslice, nk]


def krange_from_slice(kslice, grid):
    kstart = kslice.start
    kend = kslice.stop
    nk = grid.npz - kstart if kend is None else kend - kstart
    return kstart, nk


def great_circle_dist(p1, p2, radius=None):
    beta = (
        math.asin(
            math.sqrt(
                math.sin((p1[1] - p2[1]) / 2.0) ** 2
                + math.cos(p1[1])
                * math.cos(p2[1])
                * math.sin((p1[0] - p2[0]) / 2.0) ** 2
            )
        )
        * 2.0
    )
    if radius is not None:
        great_circle_dist = radius * beta
    else:
        great_circle_dist = beta
    return great_circle_dist


def extrap_corner(p0, p1, p2, q1, q2):
    x1 = great_circle_dist(p1, p0)
    x2 = great_circle_dist(p2, p0)
    return q1 + x1 / (x2 - x1) * (q1 - q2)


def asarray(array, to_type=np.ndarray, dtype=None, order=None):
    if cp and (
        isinstance(array, memoryview)
        or isinstance(array.data, cp.ndarray)
        or isinstance(array.data, cp.cuda.memory.MemoryPointer)
    ):
        if to_type is np.ndarray:
            order = "F" if order is None else order
            return cp.asnumpy(array, order=order)
        else:
            return cp.asarray(array, dtype, order)
    else:
        if to_type is np.ndarray:
            return np.asarray(array, dtype, order)
        else:
            return cp.asarray(array, dtype, order)


def zeros(shape, storage_type=np.ndarray, dtype=_dtype, order="F"):
    xp = cp if cp and storage_type is cp.ndarray else np
    return xp.zeros(shape)


def sum(array, axis=None, dtype=None, out=None, keepdims=False):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.sum(array, axis, dtype, out, keepdims)


def repeat(array, repeats, axis=None):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.repeat(array.data, repeats, axis)


def index(array, key):
    return asarray(array, type(key))[key]
