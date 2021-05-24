import functools
from typing import Iterable, Tuple, Union

import numpy as np
from gt4py import gtscript

import fv3core.utils.global_config as global_config
import fv3gfs.util as fv3util

from . import gt4py_utils as utils
from .typing import Index3D


class Grid:
    # indices = ["is_", "ie", "isd", "ied", "js", "je", "jsd", "jed"]
    index_pairs = [("is_", "js"), ("ie", "je"), ("isd", "jsd"), ("ied", "jed")]
    shape_params = ["npz", "npx", "npy"]
    # npx -- number of grid corners on one tile of the domain
    # grid.ie == npx - 1identified east edge in fortran
    # But we need to add the halo - 1 to change this check to 0 based python arrays
    # grid.ie == npx + halo - 2

    def __init__(self, indices, shape_params, rank, layout, data_fields={}):
        self.rank = rank
        self.partitioner = fv3util.TilePartitioner(layout)
        self.subtile_index = self.partitioner.subtile_index(self.rank)
        self.layout = layout
        for s in self.shape_params:
            setattr(self, s, int(shape_params[s]))
        self.subtile_width_x = int((self.npx - 1) / self.layout[0])
        self.subtile_width_y = int((self.npy - 1) / self.layout[1])
        for ivar, jvar in self.index_pairs:
            local_i, local_j = self.global_to_local_indices(
                int(indices[ivar]), int(indices[jvar])
            )
            setattr(self, ivar, local_i)
            setattr(self, jvar, local_j)
        self.nid = int(self.ied - self.isd + 1)
        self.njd = int(self.jed - self.jsd + 1)
        self.nic = int(self.ie - self.is_ + 1)
        self.njc = int(self.je - self.js + 1)
        self.halo = utils.halo
        self.global_is, self.global_js = self.local_to_global_indices(self.is_, self.js)
        self.global_ie, self.global_je = self.local_to_global_indices(self.ie, self.je)
        self.global_isd, self.global_jsd = self.local_to_global_indices(
            self.isd, self.jsd
        )
        self.global_ied, self.global_jed = self.local_to_global_indices(
            self.ied, self.jed
        )
        self.west_edge = self.global_is == self.halo
        self.east_edge = self.global_ie == self.npx + self.halo - 2
        self.south_edge = self.global_js == self.halo
        self.north_edge = self.global_je == self.npy + self.halo - 2

        self.j_offset = self.js - self.jsd - 1
        self.i_offset = self.is_ - self.isd - 1
        self.sw_corner = self.west_edge and self.south_edge
        self.se_corner = self.east_edge and self.south_edge
        self.nw_corner = self.west_edge and self.north_edge
        self.ne_corner = self.east_edge and self.north_edge
        self.data_fields = {}
        self.add_data(data_fields)
        self._sizer = None
        self._quantity_factory = None

    @property
    def sizer(self):
        if self._sizer is None:
            # in the future this should use from_namelist, when we have a non-flattened
            # namelist
            self._sizer = fv3util.SubtileGridSizer.from_tile_params(
                nx_tile=self.npx - 1,
                ny_tile=self.npy - 1,
                nz=self.npz,
                n_halo=self.halo,
                extra_dim_lengths={},
                layout=self.layout,
            )
        return self._sizer

    @property
    def quantity_factory(self):
        if self._quantity_factory is None:
            self._quantity_factory = fv3util.QuantityFactory.from_backend(
                self.sizer, backend=global_config.get_backend()
            )
        return self._quantity_factory

    def make_quantity(
        self,
        array,
        dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
        units="Unknown",
        origin=None,
        extent=None,
    ):
        if origin is None:
            origin = self.compute_origin()
        if extent is None:
            extent = self.domain_shape_compute()
        return fv3util.Quantity(
            array, dims=dims, units=units, origin=origin, extent=extent
        )

    def quantity_dict_update(
        self,
        data_dict,
        varname,
        dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
        units="Unknown",
    ):
        data_dict[varname + "_quantity"] = self.quantity_wrap(
            data_dict[varname], dims=dims, units=units
        )

    def quantity_wrap(
        self, data, dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], units="Unknown"
    ):
        origin = self.sizer.get_origin(dims)
        extent = self.sizer.get_extent(dims)
        return fv3util.Quantity(
            data, dims=dims, units=units, origin=origin, extent=extent
        )

    def global_to_local_1d(self, global_value, subtile_index, subtile_length):
        return global_value - subtile_index * subtile_length

    def global_to_local_x(self, i_global):
        return self.global_to_local_1d(
            i_global, self.subtile_index[1], self.subtile_width_x
        )

    def global_to_local_y(self, j_global):
        return self.global_to_local_1d(
            j_global, self.subtile_index[0], self.subtile_width_y
        )

    def global_to_local_indices(self, i_global, j_global):
        i_local = self.global_to_local_x(i_global)
        j_local = self.global_to_local_y(j_global)
        return i_local, j_local

    def local_to_global_1d(self, local_value, subtile_index, subtile_length):
        return local_value + subtile_index * subtile_length

    def local_to_global_indices(self, i_local, j_local):
        i_global = self.local_to_global_1d(
            i_local, self.subtile_index[1], self.subtile_width_x
        )
        j_global = self.local_to_global_1d(
            j_local, self.subtile_index[0], self.subtile_width_y
        )
        return i_global, j_global

    def add_data(self, data_dict):
        self.data_fields.update(data_dict)
        for k, v in self.data_fields.items():
            setattr(self, k, v)

    def irange_compute(self):
        return range(self.is_, self.ie + 1)

    def irange_compute_x(self):
        return range(self.is_, self.ie + 2)

    def jrange_compute(self):
        return range(self.js, self.je + 1)

    def jrange_compute_y(self):
        return range(self.js, self.je + 2)

    def irange_domain(self):
        return range(self.isd, self.ied + 1)

    def jrange_domain(self):
        return range(self.jsd, self.jed + 1)

    def krange(self):
        return range(0, self.npz)

    def compute_interface(self):
        return self.slice_dict(self.compute_dict())

    def x3d_interface(self):
        return self.slice_dict(self.x3d_compute_dict())

    def y3d_interface(self):
        return self.slice_dict(self.y3d_compute_dict())

    def x3d_domain_interface(self):
        return self.slice_dict(self.x3d_domain_dict())

    def y3d_domain_interface(self):
        return self.slice_dict(self.y3d_domain_dict())

    def add_one(self, num):
        if num is None:
            return None
        return num + 1

    def slice_dict(self, d, ndim: int = 3):
        iters: str = "ijk" if ndim > 1 else "k"
        return tuple(
            [
                slice(d[f"{iters[i]}start"], self.add_one(d[f"{iters[i]}end"]))
                for i in range(ndim)
            ]
        )

    def default_domain_dict(self):
        return {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.jsd,
            "jend": self.jed,
            "kstart": 0,
            "kend": self.npz - 1,
        }

    def default_dict_buffer_2d(self):
        mydict = self.default_domain_dict()
        mydict["iend"] += 1
        mydict["jend"] += 1
        return mydict

    def compute_dict(self):
        return {
            "istart": self.is_,
            "iend": self.ie,
            "jstart": self.js,
            "jend": self.je,
            "kstart": 0,
            "kend": self.npz - 1,
        }

    def compute_dict_buffer_2d(self):
        mydict = self.compute_dict()
        mydict["iend"] += 1
        mydict["jend"] += 1
        return mydict

    def default_buffer_k_dict(self):
        mydict = self.default_domain_dict()
        mydict["kend"] = self.npz
        return mydict

    def compute_buffer_k_dict(self):
        mydict = self.compute_dict()
        mydict["kend"] = self.npz
        return mydict

    def x3d_domain_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied + 1,
            "jstart": self.jsd,
            "jend": self.jed,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_domain_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.jsd,
            "jend": self.jed + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def x3d_compute_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie + 1,
            "jstart": self.js,
            "jend": self.je,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_compute_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie,
            "jstart": self.js,
            "jend": self.je + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def x3d_compute_domain_y_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie + 1,
            "jstart": self.jsd,
            "jend": self.jed,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_compute_domain_x_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.js,
            "jend": self.je + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def domain_shape_full(self, *, add: Tuple[int, int, int] = (0, 0, 0)):
        """Domain shape for the full array including halo points."""
        return (self.nid + add[0], self.njd + add[1], self.npz + add[2])

    def domain_shape_compute(self, *, add: Tuple[int, int, int] = (0, 0, 0)):
        """Compute domain shape excluding halo points."""
        return (self.nic + add[0], self.njc + add[1], self.npz + add[2])

    def copy_right_edge(self, var, i_index, j_index):
        return np.copy(var[i_index:, :, :]), np.copy(var[:, j_index:, :])

    def insert_left_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        if len(var.shape) < 3:
            var[:i_index, :] = edge_data_i
            var[:, :j_index] = edge_data_j
        else:
            var[:i_index, :, :] = edge_data_i
            var[:, :j_index, :] = edge_data_j

    def insert_right_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        if len(var.shape) < 3:
            var[i_index:, :] = edge_data_i
            var[:, j_index:] = edge_data_j
        else:
            var[i_index:, :, :] = edge_data_i
            var[:, j_index:, :] = edge_data_j

    def uvar_edge_halo(self, var):
        return self.copy_right_edge(var, self.ie + 2, self.je + 1)

    def vvar_edge_halo(self, var):
        return self.copy_right_edge(var, self.ie + 1, self.je + 2)

    def compute_origin(self, add: Tuple[int, int, int] = (0, 0, 0)):
        """Start of the compute domain (e.g. (halo, halo, 0))"""
        return (self.is_ + add[0], self.js + add[1], add[2])

    def full_origin(self, add: Tuple[int, int, int] = (0, 0, 0)):
        """Start of the full array including halo points (e.g. (0, 0, 0))"""
        return (self.isd + add[0], self.jsd + add[1], add[2])

    def default_origin(self):
        # This only exists as a reminder because devs might
        # be used to writing "default origin"
        # if it's no longer useful please delete this method
        raise NotImplementedError(
            "This has been renamed to `full_origin`, update your code!"
        )

    # TODO, expand to more cases
    def horizontal_starts_from_shape(self, shape):
        if shape[0:2] in [
            self.domain_shape_compute()[0:2],
            self.domain_shape_compute(add=(1, 0, 0))[0:2],
            self.domain_shape_compute(add=(0, 1, 0))[0:2],
            self.domain_shape_compute(add=(1, 1, 0))[0:2],
        ]:
            return self.is_, self.js
        elif shape[0:2] == (self.nic + 2, self.njc + 2):
            return self.is_ - 1, self.js - 1
        else:
            return 0, 0

    @property
    def grid_indexing(self) -> "GridIndexing":
        return GridIndexing.from_legacy_grid(self)


class GridIndexing:
    """
    Provides indices for cell-centered variables with halos.

    These indices can be used with horizontal interface variables by adding 1
    to the domain shape along any interface axis.
    """

    def __init__(
        self,
        origin: Index3D,
        domain: Index3D,
        n_halo: int,
        south_edge: bool,
        north_edge: bool,
        west_edge: bool,
        east_edge: bool,
    ):
        self.origin = origin
        self.domain = domain
        self.n_halo = n_halo
        # TODO: split edge booleans into their own class, they aren't needed
        # for any of the indexing operations, only needed for axis_offsets
        # maybe give this responsibility to CubedSphereCommunicator.
        self.south_edge = south_edge
        self.north_edge = north_edge
        self.west_edge = west_edge
        self.east_edge = east_edge

    @classmethod
    def from_sizer_and_communicator(
        cls, sizer: fv3util.GridSizer, cube: fv3util.CubedSphereCommunicator
    ) -> "GridIndexing":
        origin = sizer.get_origin([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM])
        domain = sizer.get_extent([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM])
        south_edge = cube.tile.on_tile_bottom(cube.rank)
        north_edge = cube.tile.on_tile_top(cube.rank)
        west_edge = cube.tile.on_tile_left(cube.rank)
        east_edge = cube.tile.on_tile_right(cube.rank)
        return cls(
            origin=origin,
            domain=domain,
            n_halo=sizer.n_halo,
            south_edge=south_edge,
            north_edge=north_edge,
            west_edge=west_edge,
            east_edge=east_edge,
        )

    @classmethod
    def from_legacy_grid(cls, grid: Grid) -> "GridIndexing":
        return cls(
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(),
            n_halo=grid.halo,
            south_edge=grid.south_edge,
            north_edge=grid.north_edge,
            west_edge=grid.west_edge,
            east_edge=grid.east_edge,
        )

    @property
    def isc(self):
        """start of the compute domain along the x-axis"""
        return self.origin[0]

    @property
    def iec(self):
        """end of the compute domain along the x-axis"""
        return self.origin[0] + self.domain[0]

    @property
    def jsc(self):
        """start of the compute domain along the y-axis"""
        return self.origin[1]

    @property
    def jec(self):
        """end of the compute domain along the y-axis"""
        return self.origin[1] + self.domain[1]

    @property
    def ks(self):
        """start of the data domain along the z-axis"""
        return self.origin[2]

    @property
    def ke(self):
        """end of the data domain along the z-axis"""
        return self.origin[2] + self.domain[2]

    @property
    def isd(self):
        """start of the full domain including halos along the x-axis"""
        return self.origin[0] - self.n_halo

    @property
    def ied(self):
        """end of the full domain including halos along the x-axis"""
        return self.isd + self.domain[0] + 2 * self.n_halo

    @property
    def jsd(self):
        """start of the full domain including halos along the y-axis"""
        return self.origin[1] - self.n_halo

    @property
    def jed(self):
        """end of the full domain including halos along the y-axis"""
        return self.jsd + self.domain[1] + 2 * self.n_halo

    def origin_full(self, add: Index3D = (0, 0, 0)):
        """
        Returns the origin of the full domain including halos, plus an optional offset.
        """
        return (self.isd + add[0], self.jsd + add[1], self.ks + add[2])

    def origin_compute(self, add: Index3D = (0, 0, 0)):
        """
        Returns the origin of the compute domain, plus an optional offset
        """
        return (self.isc + add[0], self.jsc + add[1], self.ks + add[2])

    def domain_full(self, add: Index3D = (0, 0, 0)):
        """
        Returns the shape of the full domain including halos, plus an optional offset.
        """
        return (
            self.ied - self.isd + add[0],
            self.jed - self.jsd + add[1],
            self.ke - self.ks + add[2],
        )

    def domain_compute(self, add: Index3D = (0, 0, 0)):
        """
        Returns the shape of the compute domain, plus an optional offset.
        """
        return (
            self.iec - self.isc + add[0],
            self.jec - self.jsc + add[1],
            self.ke - self.ks + add[2],
        )


def axis_offsets(
    grid: Union[Grid, GridIndexing],
    origin: Iterable[int],
    domain: Iterable[int],
):
    """Return the axis offsets relative to stencil compute domain."""
    origin = tuple(origin)
    domain = tuple(domain)
    if isinstance(grid, Grid):
        return _old_grid_axis_offsets(grid, origin, domain)
    else:
        return _grid_indexing_axis_offsets(grid, origin, domain)


@functools.lru_cache(maxsize=None)
def _old_grid_axis_offsets(
    grid: Grid,
    origin: Tuple[int, ...],
    domain: Tuple[int, ...],
):
    if grid.west_edge:
        proc_offset = grid.is_ - grid.global_is
        origin_offset = grid.is_ - origin[0]
        i_start = gtscript.I[0] + proc_offset + origin_offset
    else:
        i_start = gtscript.I[0] - np.iinfo(np.int32).max

    if grid.east_edge:
        proc_offset = grid.npx + grid.halo - 2 - grid.global_is
        endpt_offset = (grid.is_ - origin[0]) - domain[0] + 1
        i_end = gtscript.I[-1] + proc_offset + endpt_offset
    else:
        i_end = gtscript.I[-1] + np.iinfo(np.int32).max

    if grid.south_edge:
        proc_offset = grid.js - grid.global_js
        origin_offset = grid.js - origin[1]
        j_start = gtscript.J[0] + proc_offset + origin_offset
    else:
        j_start = gtscript.J[0] - np.iinfo(np.int32).max

    if grid.north_edge:
        proc_offset = grid.npy + grid.halo - 2 - grid.global_js
        endpt_offset = (grid.js - origin[1]) - domain[1] + 1
        j_end = gtscript.J[-1] + proc_offset + endpt_offset
    else:
        j_end = gtscript.J[-1] + np.iinfo(np.int32).max

    return {
        "i_start": i_start,
        "local_is": gtscript.I[0] + grid.is_ - origin[0],
        "i_end": i_end,
        "local_ie": gtscript.I[-1] + grid.ie - origin[0] - domain[0] + 1,
        "j_start": j_start,
        "local_js": gtscript.J[0] + grid.js - origin[1],
        "j_end": j_end,
        "local_je": gtscript.J[-1] + grid.je - origin[1] - domain[1] + 1,
    }


@functools.lru_cache(maxsize=None)
def _grid_indexing_axis_offsets(
    grid: GridIndexing,
    origin: Tuple[int, ...],
    domain: Tuple[int, ...],
):
    if grid.west_edge:
        i_start = gtscript.I[0] + grid.origin[0] - origin[0]
    else:
        i_start = gtscript.I[0] - np.iinfo(np.int32).max

    if grid.east_edge:
        i_end = (
            gtscript.I[-1] + (grid.origin[0] + grid.domain[0]) - (origin[0] + domain[0])
        )
    else:
        i_end = gtscript.I[-1] + np.iinfo(np.int32).max

    if grid.south_edge:
        j_start = gtscript.J[0] + grid.origin[1] - origin[1]
    else:
        j_start = gtscript.J[0] - np.iinfo(np.int32).max

    if grid.north_edge:
        j_end = (
            gtscript.J[-1] + (grid.origin[1] + grid.domain[1]) - (origin[1] + domain[1])
        )
    else:
        j_end = gtscript.J[-1] + np.iinfo(np.int32).max

    return {
        "i_start": i_start,
        "local_is": gtscript.I[0] + grid.isc - origin[0],
        "i_end": i_end,
        "local_ie": gtscript.I[-1] + grid.iec - origin[0] - domain[0] + 1,
        "j_start": j_start,
        "local_js": gtscript.J[0] + grid.jsc - origin[1],
        "j_end": j_end,
        "local_je": gtscript.J[-1] + grid.jec - origin[1] - domain[1] + 1,
    }
