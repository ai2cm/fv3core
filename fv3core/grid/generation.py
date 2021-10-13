import functools

import fv3gfs.util as fv3util
from fv3core.utils.corners import (
    fill_corners_2d,
    fill_corners_agrid,
    fill_corners_cgrid,
    fill_corners_dgrid,
)
from fv3core.utils.global_constants import CARTESIAN_DIM, LON_OR_LAT_DIM, PI, RADIUS
from fv3core.utils.grid import GridIndexing
from fv3gfs.util.constants import N_HALO_DEFAULT

from .eta import set_eta
from .geometry import (
    calc_unit_vector_south,
    calc_unit_vector_west,
    calculate_divg_del6,
    calculate_grid_a,
    calculate_grid_z,
    calculate_l2c_vu,
    calculate_supergrid_cos_sin,
    calculate_trig_uv,
    edge_factors,
    efactor_a2c_v,
    generate_xy_unit_vectors,
    get_center_vector,
    supergrid_corner_fix,
    unit_vector_lonlat,
)
from .gnomonic import (
    get_area,
    great_circle_distance_along_axis,
    local_gnomonic_ed,
    lon_lat_corner_to_cell_center,
    lon_lat_midpoint,
    lon_lat_to_xyz,
    set_c_grid_tile_border_area,
    set_corner_area_to_triangle_area,
    set_tile_border_dxc,
    set_tile_border_dyc,
)
from .mirror import mirror_grid


# TODO remove this when using python 3.8+ everywhere, it comes for free
def cached_property(func):
    cached = None

    @property
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        nonlocal cached
        if cached is None:
            cached = func(*args, **kwargs)
        return cached

    return wrapped


# TODO
# corners use sizer + partitioner rather than GridIndexer,
# requires fv3core clls to corners know what to do
class MetricTerms:
    def __init__(
        self,
        *,
        quantity_factory: fv3util.QuantityFactory,
        communicator: fv3util.Communicator,
        grid_type: int = 0,
    ):
        assert grid_type < 3
        self._grid_type = grid_type
        self._halo = N_HALO_DEFAULT
        self._comm = communicator
        self._partitioner = self._comm.partitioner
        self._tile_partitioner = self._partitioner.tile
        self._rank = self._comm.rank
        self._quantity_factory = quantity_factory
        self._grid_indexer = GridIndexing.from_sizer_and_communicator(
            self._quantity_factory._sizer, self._comm
        )
        self._grid_dims = [
            fv3util.X_INTERFACE_DIM,
            fv3util.Y_INTERFACE_DIM,
            LON_OR_LAT_DIM,
        ]
        self._grid = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            "radians",
            dtype=float,
        )
        npx, npy, ndims = self._tile_partitioner.global_extent(self._grid)
        self._npx = npx
        self._npy = npy
        self._npz = self._quantity_factory._sizer.get_extent(fv3util.Z_DIM)[0]
        self._agrid = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM], "radians", dtype=float
        )
        self._np = self._grid.np
        self._dx = None
        self._dy = None
        self._dx_agrid = None
        self._dy_agrid = None
        self._dx_cgrid = None
        self._dy_cgrid = None
        self._ak = None
        self._bk = None
        self._ks = None
        self._ptop = None
        self._ec1 = None
        self._ec2 = None
        self._ew1 = None
        self._ew2 = None
        self._es1 = None
        self._es2 = None
        self._ee1 = None
        self._ee2 = None
        self._l2c_v = None
        self._l2c_u = None
        self._cos_sg1 = None
        self._cos_sg2 = None
        self._cos_sg3 = None
        self._cos_sg4 = None
        self._cos_sg5 = None
        self._cos_sg6 = None
        self._cos_sg7 = None
        self._cos_sg8 = None
        self._cos_sg9 = None
        self._sin_sg1 = None
        self._sin_sg2 = None
        self._sin_sg3 = None
        self._sin_sg4 = None
        self._sin_sg5 = None
        self._sin_sg6 = None
        self._sin_sg7 = None
        self._sin_sg8 = None
        self._sin_sg9 = None
        self._cosa = None
        self._sina = None
        self._cosa_u = None
        self._cosa_v = None
        self._cosa_s = None
        self._sina_u = None
        self._sina_v = None
        self._rsin_u = None
        self._rsin_v = None
        self._rsina = None
        self._rsin2 = None
        self._del6_u = None
        self._del6_v = None
        self._divg_u = None
        self._divg_v = None
        self._vlon = None
        self._vlat = None
        self._z11 = None
        self._z12 = None
        self._z21 = None
        self._z22 = None
        self._a11 = None
        self._a12 = None
        self._a21 = None
        self._a22 = None
        self._edge_w = None
        self._edge_e = None
        self._edge_s = None
        self._edge_n = None
        self._edge_vect_w = None
        self._edge_vect_e = None
        self._edge_vect_s = None
        self._edge_vect_n = None
        self._da_min = None
        self._da_max = None
        self._da_min_c = None
        self._da_max_c = None

        self._init_dgrid()
        self._init_agrid()

    @classmethod
    def from_tile_sizing(
        cls,
        npx: int,
        npy: int,
        npz: int,
        communicator: fv3util.Communicator,
        backend: str,
        grid_type: int = 0,
    ) -> "MetricTerms":
        sizer = fv3util.SubtileGridSizer.from_tile_params(
            nx_tile=npx - 1,
            ny_tile=npy - 1,
            nz=npz,
            n_halo=N_HALO_DEFAULT,
            extra_dim_lengths={
                LON_OR_LAT_DIM: 2,
                CARTESIAN_DIM: 3,
            },
            layout=communicator.partitioner.tile.layout,
        )
        quantity_factory = fv3util.QuantityFactory.from_backend(sizer, backend=backend)
        return cls(
            quantity_factory=quantity_factory,
            communicator=communicator,
            grid_type=grid_type,
        )
    
    @property
    def grid(self):
        return self._grid
    
    @property
    def gridvar(self):
        return self._grid

    @property
    def agrid(self):
        return self._agrid

    @property
    def dx(self):
        if self._dx is None:
            self._dx, self._dy = self._compute_dxdy()
        return self._dx

    @property
    def dy(self):
        if self._dy is None:
            self._dx, self._dy = self._compute_dxdy()
        return self._dy

    @property
    def dxa(self):
        if self._dx_agrid is None:
            self._dx_agrid, self._dy_agrid = self._compute_dxdy_agrid()
        return self._dx_agrid

    @property
    def dya(self):
        if self._dy_agrid is None:
            self._dx_agrid, self._dy_agrid = self._compute_dxdy_agrid()
        return self._dy_agrid

    @property
    def dxc(self):
        if self._dx_cgrid is None:
            self._dx_cgrid, self._dy_cgrid = self._compute_dxdy_cgrid()
        return self._dx_cgrid

    @property
    def dyc(self):
        if self._dy_cgrid is None:
            self._dx_cgrid, self._dy_cgrid = self._compute_dxdy_cgrid()
        return self._dy_cgrid

    @property
    def ak(self):
        if self._ak is None:
            self._ks, self._ptop, self._ak, self._bk = self._set_eta()
        return self._ak

    @property
    def bk(self):
        if self._ak is None:
            self._ks, self._ptop, self._ak, self._bk = self._set_eta()
        return self._bk

    @property
    def ks(self):
        if self._ak is None:
            self._ks, self._ptop, self._ak, self._bk = self._set_eta()
        return self._ks

    @property
    def ptop(self):
        if self._ak is None:
            self._ks, self._ptop, self._ak, self._bk = self._set_eta()
        return self._ptop

    @property
    def ec1(self):
        if self._ec1 is None:
            self._ec1, self._ec2 = self._calculate_center_vectors()
        return self._ec1

    @property
    def ec2(self):
        if self._ec2 is None:
            self._ec1, self._ec2 = self._calculate_center_vectors()
        return self._ec2

    @property
    def ew1(self):
        if self._ew1 is None:
            self._ew1, self._ew2 = self._calculate_vectors_west()
        return self._ew1

    @property
    def ew2(self):
        if self._ew2 is None:
            self._ew1, self._ew2 = self._calculate_vectors_west()
        return self._ew2

    @property
    def cos_sg1(self):
        if self._cos_sg1 is None:
            self._init_cell_trigonometry()
        return self._cos_sg1

    @property
    def cos_sg2(self):
        if self._cos_sg2 is None:
            self._init_cell_trigonometry()
        return self._cos_sg2

    @property
    def cos_sg3(self):
        if self._cos_sg3 is None:
            self._init_cell_trigonometry()
        return self._cos_sg3

    @property
    def cos_sg4(self):
        if self._cos_sg4 is None:
            self._init_cell_trigonometry()
        return self._cos_sg4

    @property
    def cos_sg5(self):
        if self._cos_sg5 is None:
            self._init_cell_trigonometry()
        return self._cos_sg5

    @property
    def cos_sg6(self):
        if self._cos_sg6 is None:
            self._init_cell_trigonometry()
        return self._cos_sg6

    @property
    def cos_sg7(self):
        if self._cos_sg7 is None:
            self._init_cell_trigonometry()
        return self._cos_sg7

    @property
    def cos_sg8(self):
        if self._cos_sg8 is None:
            self._init_cell_trigonometry()
        return self._cos_sg8

    @property
    def cos_sg9(self):
        if self._cos_sg9 is None:
            self._init_cell_trigonometry()
        return self._cos_sg9

    @property
    def sin_sg1(self):
        if self._sin_sg1 is None:
            self._init_cell_trigonometry()
        return self._sin_sg1

    @property
    def sin_sg2(self):
        if self._sin_sg2 is None:
            self._init_cell_trigonometry()
        return self._sin_sg2

    @property
    def sin_sg3(self):
        if self._sin_sg3 is None:
            self._init_cell_trigonometry()
        return self._sin_sg3

    @property
    def sin_sg4(self):
        if self._sin_sg4 is None:
            self._init_cell_trigonometry()
        return self._sin_sg4

    @property
    def sin_sg5(self):
        if self._sin_sg5 is None:
            self._init_cell_trigonometry()
        return self._sin_sg5

    @property
    def sin_sg6(self):
        if self._sin_sg6 is None:
            self._init_cell_trigonometry()
        return self._sin_sg6

    @property
    def sin_sg7(self):
        if self._sin_sg7 is None:
            self._init_cell_trigonometry()
        return self._sin_sg7

    @property
    def sin_sg8(self):
        if self._sin_sg8 is None:
            self._init_cell_trigonometry()
        return self._sin_sg8

    @property
    def sin_sg9(self):
        if self._sin_sg9 is None:
            self._init_cell_trigonometry()
        return self._sin_sg9

    @property
    def cosa(self):
        if self._cosa is None:
            self._init_cell_trigonometry()
        return self._cosa

    @property
    def sina(self):
        if self._sina is None:
            self._init_cell_trigonometry()
        return self._sina

    @property
    def cosa_u(self):
        if self._cosa_u is None:
            self._init_cell_trigonometry()
        return self._cosa_u

    @property
    def cosa_v(self):
        if self._cosa_v is None:
            self._init_cell_trigonometry()
        return self._cosa_v

    @property
    def cosa_s(self):
        if self._cosa_s is None:
            self._init_cell_trigonometry()
        return self._cosa_s

    @property
    def sina_u(self):
        if self._sina_u is None:
            self._init_cell_trigonometry()
        return self._sina_u

    @property
    def sina_v(self):
        if self._sina_v is None:
            self._init_cell_trigonometry()
        return self._sina_v

    @property
    def rsin_u(self):
        if self._rsin_u is None:
            self._init_cell_trigonometry()
        return self._rsin_u

    @property
    def rsin_v(self):
        if self._rsin_v is None:
            self._init_cell_trigonometry()
        return self._rsin_v

    @property
    def rsina(self):
        if self._rsina is None:
            self._init_cell_trigonometry()
        return self._rsina

    @property
    def rsin2(self):
        if self._rsin2 is None:
            self._init_cell_trigonometry()
        return self._rsin2

    @property
    def l2c_v(self):
        if self._l2c_v is None:
            self._l2c_v, self._l2c_u = self._calculate_latlon_momentum_correction()
        return self._l2c_v

    @property
    def l2c_u(self):
        if self._l2c_u is None:
            self._l2c_v, self._l2c_u = self._calculate_latlon_momentum_correction()
        return self._l2c_u

    @property
    def es1(self):
        if self._es1 is None:
            self._es1, self._es2 = self._calculate_vectors_south()
        return self._es1

    @property
    def es2(self):
        if self._es2 is None:
            self._es1, self._es2 = self._calculate_vectors_south()
        return self._es2

    @property
    def ee1(self):
        if self._ee1 is None:
            self._ee1, self._ee2 = self._calculate_xy_unit_vectors()
        return self._ee1

    @property
    def ee2(self):
        if self._ee2 is None:
            self._ee1, self._ee2 = self._calculate_xy_unit_vectors()
        return self._ee2

    @property
    def divg_u(self):
        if self._divg_u is None:
            (
                self._del6_u,
                self._del6_v,
                self._divg_u,
                self._divg_v,
            ) = self._calculate_divg_del6()
        return self._divg_u

    @property
    def divg_v(self):
        if self._divg_v is None:
            (
                self._del6_u,
                self._del6_v,
                self._divg_u,
                self._divg_v,
            ) = self._calculate_divg_del6()
        return self._divg_v

    @property
    def del6_u(self):
        if self._del6_u is None:
            (
                self._del6_u,
                self._del6_v,
                self._divg_u,
                self._divg_v,
            ) = self._calculate_divg_del6()
        return self._del6_u

    @property
    def del6_v(self):
        if self._del6_v is None:
            (
                self._del6_u,
                self._del6_v,
                self._divg_u,
                self._divg_v,
            ) = self._calculate_divg_del6()
        return self._del6_v

    @property
    def vlon(self):
        if self._vlon is None:
            self._vlon, self._vlat = self._calculate_unit_vectors_lonlat()
        return self._vlon

    @property
    def vlat(self):
        if self._vlat is None:
            self._vlon, self._vlat = self._calculate_unit_vectors_lonlat()
        return self._vlat

    @property
    def z11(self):
        if self._z11 is None:
            self._z11, self._z12, self._z21, self._z22 = self._calculate_grid_z()
        return self._z11

    @property
    def z12(self):
        if self._z12 is None:
            self._z11, self._z12, self._z21, self._z22 = self._calculate_grid_z()
        return self._z12

    @property
    def z21(self):
        if self._z21 is None:
            self._z11, self._z12, self._z21, self._z22 = self._calculate_grid_z()
        return self._z21

    @property
    def z22(self):
        if self._z22 is None:
            self._z11, self._z12, self._z21, self._z22 = self._calculate_grid_z()
        return self._z22

    @property
    def a11(self):
        if self._a11 is None:
            self._a11, self._a12, self._a21, self._a22 = self._calculate_grid_a()
        return self._a11

    @property
    def a12(self):
        if self._a12 is None:
            self._a11, self._a12, self._a21, self._a22 = self._calculate_grid_a()
        return self._a12

    @property
    def a21(self):
        if self._a21 is None:
            self._a11, self._a12, self._a21, self._a22 = self._calculate_grid_a()
        return self._a21

    @property
    def a22(self):
        if self._a22 is None:
            self._a11, self._a12, self._a21, self._a22 = self._calculate_grid_a()
        return self._a22

    @property
    def edge_w(self):
        if self._edge_w is None:
            (
                self._edge_w,
                self._edge_e,
                self._edge_s,
                self._edge_n,
            ) = self._calculate_edge_factors()
        return self._edge_w

    @property
    def edge_e(self):
        if self._edge_e is None:
            (
                self._edge_w,
                self._edge_e,
                self._edge_s,
                self._edge_n,
            ) = self._calculate_edge_factors()
        return self._edge_e

    @property
    def edge_s(self):
        if self._edge_s is None:
            (
                self._edge_w,
                self._edge_e,
                self._edge_s,
                self._edge_n,
            ) = self._calculate_edge_factors()
        return self._edge_s

    @property
    def edge_n(self):
        if self._edge_n is None:
            (
                self._edge_w,
                self._edge_e,
                self._edge_s,
                self._edge_n,
            ) = self._calculate_edge_factors()
        return self._edge_n

    @property
    def edge_vect_w(self):
        if self._edge_vect_w is None:
            (
                self._edge_vect_w,
                self._edge_vect_e,
                self._edge_vect_s,
                self._edge_vect_n,
            ) = self._calculate_edge_a2c_vect_factors()
        return self._edge_vect_w

    @property
    def edge_vect_e(self):
        if self._edge_vect_e is None:
            (
                self._edge_vect_w,
                self._edge_vect_e,
                self._edge_vect_s,
                self._edge_vect_n,
            ) = self._calculate_edge_a2c_vect_factors()
        return self._edge_vect_e

    @property
    def edge_vect_s(self):
        if self._edge_vect_s is None:
            (
                self._edge_vect_w,
                self._edge_vect_e,
                self._edge_vect_s,
                self._edge_vect_n,
            ) = self._calculate_edge_a2c_vect_factors()
        return self._edge_vect_s

    @property
    def edge_vect_n(self):
        if self._edge_vect_n is None:
            (
                self._edge_vect_w,
                self._edge_vect_e,
                self._edge_vect_s,
                self._edge_vect_n,
            ) = self._calculate_edge_a2c_vect_factors()
        return self._edge_vect_n

    @property
    def da_min(self):
        if self._da_min is None:
            self._reduce_global_area_minmaxes()
        return self._da_min

    @property
    def da_max(self):
        if self._da_max is None:
            self._reduce_global_area_minmaxes()
        return self._da_max

    @property
    def da_min_c(self):
        if self._da_min_c is None:
            self._reduce_global_area_minmaxes()
        return self._da_min_c

    @property
    def da_max_c(self):
        if self._da_max_c is None:
            self._reduce_global_area_minmaxes()
        return self._da_max_c

    @cached_property
    def area(self):
        return self._compute_area()

    @cached_property
    def area_c(self):
        return self._compute_area_c()

    @cached_property
    def _dgrid_xyz(self):
        return lon_lat_to_xyz(
            self._grid.data[:, :, 0], self._grid.data[:, :, 1], self._np
        )

    @cached_property
    def _agrid_xyz(self):
        return lon_lat_to_xyz(
            self._agrid.data[:-1, :-1, 0],
            self._agrid.data[:-1, :-1, 1],
            self._np,
        )

    @cached_property
    def rarea(self):
        return 1.0 / self.area

    @cached_property
    def rarea_c(self):
        return 1.0 / self.area_c

    @cached_property
    def rdx(self):
        return 1.0 / self.dx

    @cached_property
    def rdy(self):
        return 1.0 / self.dy

    @cached_property
    def rdxa(self):
        return 1.0 / self.dxa

    @cached_property
    def rdya(self):
        return 1.0 / self.dya

    @cached_property
    def rdxc(self):
        return 1.0 / self.dxc

    @cached_property
    def rdyc(self):
        return 1.0 / self.dyc

    def _init_dgrid(self):

        grid_mirror_ew = self._quantity_factory.zeros(
            self._grid_dims,
            "radians",
            dtype=float,
        )
        grid_mirror_ns = self._quantity_factory.zeros(
            self._grid_dims,
            "radians",
            dtype=float,
        )
        grid_mirror_diag = self._quantity_factory.zeros(
            self._grid_dims,
            "radians",
            dtype=float,
        )

        local_west_edge = self._tile_partitioner.on_tile_left(self._rank)
        local_east_edge = self._tile_partitioner.on_tile_right(self._rank)
        local_south_edge = self._tile_partitioner.on_tile_bottom(self._rank)
        local_north_edge = self._tile_partitioner.on_tile_top(self._rank)
        # information on position of subtile in full tile
        slice_x, slice_y = self._tile_partitioner.subtile_slice(
            self._rank, self._grid.dims, (self._npx, self._npy), overlap=True
        )
        section_global_is = self._halo + slice_x.start
        section_global_js = self._halo + slice_y.start
        subtile_width_x = slice_x.stop - slice_x.start - 1
        subtile_width_y = slice_y.stop - slice_y.start - 1

        # compute gnomonic grid for this rank
        local_gnomonic_ed(
            self._grid.view[:, :, 0],
            self._grid.view[:, :, 1],
            npx=self._npx,
            west_edge=local_west_edge,
            east_edge=local_east_edge,
            south_edge=local_south_edge,
            north_edge=local_north_edge,
            global_is=section_global_is,
            global_js=section_global_js,
            np=self._np,
            rank=self._rank,
        )

        # Next compute gnomonic for the mirrored ranks that'll be averaged
        j_subtile_index, i_subtile_index = self._tile_partitioner.subtile_index(
            self._rank
        )
        # compute the global index starting points for the mirrored ranks
        ew_global_is = (
            self._halo
            + (self._tile_partitioner.layout[0] - i_subtile_index - 1) * subtile_width_x
        )
        ns_global_js = (
            self._halo
            + (self._tile_partitioner.layout[1] - j_subtile_index - 1) * subtile_width_y
        )

        # compute mirror in the east-west direction
        west_edge = True if local_east_edge else False
        east_edge = True if local_west_edge else False
        local_gnomonic_ed(
            grid_mirror_ew.view[:, :, 0],
            grid_mirror_ew.view[:, :, 1],
            npx=self._npx,
            west_edge=west_edge,
            east_edge=east_edge,
            south_edge=local_south_edge,
            north_edge=local_north_edge,
            global_is=ew_global_is,
            global_js=section_global_js,
            np=self._np,
            rank=self._rank,
        )

        # compute mirror in the north-south direction
        south_edge = True if local_north_edge else False
        north_edge = True if local_south_edge else False
        local_gnomonic_ed(
            grid_mirror_ns.view[:, :, 0],
            grid_mirror_ns.view[:, :, 1],
            npx=self._npx,
            west_edge=local_west_edge,
            east_edge=local_east_edge,
            south_edge=south_edge,
            north_edge=north_edge,
            global_is=section_global_is,
            global_js=ns_global_js,
            np=self._np,
            rank=self._rank,
        )

        local_gnomonic_ed(
            grid_mirror_diag.view[:, :, 0],
            grid_mirror_diag.view[:, :, 1],
            npx=self._npx,
            west_edge=west_edge,
            east_edge=east_edge,
            south_edge=south_edge,
            north_edge=north_edge,
            global_is=ew_global_is,
            global_js=ns_global_js,
            np=self._np,
            rank=self._rank,
        )

        # Average the mirrored gnomonic grids
        tile_index = self._partitioner.tile_index(self._rank)
        mirror_data = {
            "local": self._grid.data,
            "east-west": grid_mirror_ew.data,
            "north-south": grid_mirror_ns.data,
            "diagonal": grid_mirror_diag.data,
        }
        mirror_grid(
            mirror_data=mirror_data,
            tile_index=tile_index,
            npx=self._npx,
            npy=self._npy,
            x_subtile_width=subtile_width_x + 1,
            y_subtile_width=subtile_width_x + 1,
            global_is=section_global_is,
            global_js=section_global_js,
            ng=self._halo,
            np=self._grid.np,
        )
        # Shift the corner away from Japan
        # This will result in the corner close to east coast of China
        # TODO if not config.do_schmidt and config.shift_fac > 1.0e-4
        shift_fac = 18
        self._grid.view[:, :, 0] -= PI / shift_fac
        tile0_lon = self._grid.data[:, :, 0]
        tile0_lon[tile0_lon < 0] += 2 * PI
        self._grid.data[self._np.abs(self._grid.data[:]) < 1e-10] = 0.0

        self._comm.halo_update(self._grid, n_points=self._halo)

        fill_corners_2d(
            self._grid.data, self._grid_indexer, gridtype="B", direction="x"
        )

    def _init_agrid(self):
        # Set up lat-lon a-grid, calculate side lengths on a-grid
        lon_agrid, lat_agrid = lon_lat_corner_to_cell_center(
            self._grid.data[:, :, 0], self._grid.data[:, :, 1], self._np
        )
        self._agrid.data[:-1, :-1, 0], self._agrid.data[:-1, :-1, 1] = (
            lon_agrid,
            lat_agrid,
        )
        self._comm.halo_update(self._agrid, n_points=self._halo)
        fill_corners_2d(
            self._agrid.data[:, :, 0][:, :, None],
            self._grid_indexer,
            gridtype="A",
            direction="x",
        )
        fill_corners_2d(
            self._agrid.data[:, :, 1][:, :, None],
            self._grid_indexer,
            gridtype="A",
            direction="y",
        )

    def _compute_dxdy(self):
        dx = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], "m")

        dx.view[:, :] = great_circle_distance_along_axis(
            self._grid.view[:, :, 0],
            self._grid.view[:, :, 1],
            RADIUS,
            self._np,
            axis=0,
        )
        dy = self._quantity_factory.zeros([fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], "m")
        dy.view[:, :] = great_circle_distance_along_axis(
            self._grid.view[:, :, 0],
            self._grid.view[:, :, 1],
            RADIUS,
            self._np,
            axis=1,
        )
        self._comm.vector_halo_update(dx, dy, n_points=self._halo)

        # at this point the Fortran code copies in the west and east edges from
        # the halo for dy and performs a halo update,
        # to ensure dx and dy mirror across the boundary.
        # Not doing it here at the moment.
        dx.data[dx.data < 0] *= -1
        dy.data[dy.data < 0] *= -1
        fill_corners_dgrid(
            dx.data[:, :, None],
            dy.data[:, :, None],
            self._grid_indexer,
            vector=False,
        )
        return dx, dy

    def _compute_dxdy_agrid(self):

        dx_agrid = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "m")
        dy_agrid = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "m")
        lon, lat = self._grid.data[:, :, 0], self._grid.data[:, :, 1]
        lon_y_center, lat_y_center = lon_lat_midpoint(
            lon[:, :-1], lon[:, 1:], lat[:, :-1], lat[:, 1:], self._np
        )
        dx_agrid_tmp = great_circle_distance_along_axis(
            lon_y_center, lat_y_center, RADIUS, self._np, axis=0
        )
        lon_x_center, lat_x_center = lon_lat_midpoint(
            lon[:-1, :], lon[1:, :], lat[:-1, :], lat[1:, :], self._np
        )
        dy_agrid_tmp = great_circle_distance_along_axis(
            lon_x_center, lat_x_center, RADIUS, self._np, axis=1
        )
        fill_corners_agrid(
            dx_agrid_tmp[:, :, None],
            dy_agrid_tmp[:, :, None],
            self._grid_indexer,
            vector=False,
        )

        dx_agrid.data[:-1, :-1] = dx_agrid_tmp
        dy_agrid.data[:-1, :-1] = dy_agrid_tmp
        self._comm.vector_halo_update(dx_agrid, dy_agrid, n_points=self._halo)

        # at this point the Fortran code copies in the west and east edges from
        # the halo for dy and performs a halo update,
        # to ensure dx and dy mirror across the boundary.
        # Not doing it here at the moment.
        dx_agrid.data[dx_agrid.data < 0] *= -1
        dy_agrid.data[dy_agrid.data < 0] *= -1
        return dx_agrid, dy_agrid

    def _compute_dxdy_cgrid(self):
        dx_cgrid = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], "m"
        )
        dy_cgrid = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], "m"
        )

        lon_agrid, lat_agrid = (
            self._agrid.data[:-1, :-1, 0],
            self._agrid.data[:-1, :-1, 1],
        )
        dx_cgrid_tmp = great_circle_distance_along_axis(
            lon_agrid.data, lat_agrid.data, RADIUS, self._np, axis=0
        )
        dy_cgrid_tmp = great_circle_distance_along_axis(
            lon_agrid.data, lat_agrid.data, RADIUS, self._np, axis=1
        )
        # copying the second-to-last values to the last values is what the Fortran
        # code does, but is this correct/valid?
        # Maybe we want to change this to use halo updates?
        dx_cgrid.data[1:-1, :-1] = dx_cgrid_tmp
        dx_cgrid.data[0, :-1] = dx_cgrid_tmp[0, :]
        dx_cgrid.data[-1, :-1] = dx_cgrid_tmp[-1, :]

        dy_cgrid.data[:-1, 1:-1] = dy_cgrid_tmp
        dy_cgrid.data[:-1, 0] = dy_cgrid_tmp[:, 0]
        dy_cgrid.data[:-1, -1] = dy_cgrid_tmp[:, -1]

        set_tile_border_dxc(
            self._dgrid_xyz[3:-3, 3:-3, :],
            self._agrid_xyz[3:-3, 3:-3, :],
            RADIUS,
            dx_cgrid.data[3:-3, 3:-4],
            self._tile_partitioner,
            self._rank,
            self._np,
        )
        set_tile_border_dyc(
            self._dgrid_xyz[3:-3, 3:-3, :],
            self._agrid_xyz[3:-3, 3:-3, :],
            RADIUS,
            dy_cgrid.data[3:-4, 3:-3],
            self._tile_partitioner,
            self._rank,
            self._np,
        )
        self._comm.vector_halo_update(dx_cgrid, dy_cgrid, n_points=self._halo)

        # TODO: Add support for unsigned vector halo updates
        # instead of handling ad-hoc here
        dx_cgrid.data[dx_cgrid.data < 0] *= -1
        dy_cgrid.data[dy_cgrid.data < 0] *= -1

        # TODO: fix issue with interface dimensions causing validation errors
        fill_corners_cgrid(
            dx_cgrid.data[:, :, None],
            dy_cgrid.data[:, :, None],
            self._grid_indexer,
            vector=False,
        )

        return dx_cgrid, dy_cgrid

    def _compute_area(self):
        area = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "m^2")
        area.data[:, :] = -1.0e8

        area.data[3:-4, 3:-4] = get_area(
            self._grid.data[3:-3, 3:-3, 0],
            self._grid.data[3:-3, 3:-3, 1],
            RADIUS,
            self._np,
        )
        self._comm.halo_update(area, n_points=self._halo)
        return area

    def _compute_area_c(self):
        area_cgrid = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "m^2"
        )
        area_cgrid.data[3:-3, 3:-3] = get_area(
            self._agrid.data[2:-3, 2:-3, 0],
            self._agrid.data[2:-3, 2:-3, 1],
            RADIUS,
            self._np,
        )
        # TODO -- this does not seem to matter? running with or without does
        # not change whether it validates
        set_corner_area_to_triangle_area(
            lon=self._agrid.data[2:-3, 2:-3, 0],
            lat=self._agrid.data[2:-3, 2:-3, 1],
            area=area_cgrid.data[3:-3, 3:-3],
            tile_partitioner=self._tile_partitioner,
            rank=self._rank,
            radius=RADIUS,
            np=self._np,
        )

        set_c_grid_tile_border_area(
            self._dgrid_xyz[2:-2, 2:-2, :],
            self._agrid_xyz[2:-2, 2:-2, :],
            RADIUS,
            area_cgrid.data[3:-3, 3:-3],
            self._tile_partitioner,
            self._rank,
            self._np,
        )
        self._comm.halo_update(area_cgrid, n_points=self._halo)

        fill_corners_2d(
            area_cgrid.data[:, :, None],
            self._grid_indexer,
            gridtype="B",
            direction="x",
        )
        return area_cgrid

    def _set_eta(self):
        ks = self._quantity_factory.zeros([], "")
        ptop = self._quantity_factory.zeros([], "mb")
        ak = self._quantity_factory.zeros([fv3util.Z_INTERFACE_DIM], "mb")
        bk = self._quantity_factory.zeros([fv3util.Z_INTERFACE_DIM], "mb")
        ks, ptop, ak.data[:], bk.data[:] = set_eta(self._npz)
        return ks, ptop, ak, bk

    def _calculate_center_vectors(self):
        ec1 = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )
        ec2 = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )
        ec1.data[:] = self._np.nan
        ec2.data[:] = self._np.nan
        ec1.data[:-1, :-1, :3], ec2.data[:-1, :-1, :3] = get_center_vector(
            self._dgrid_xyz,
            self._grid_type,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )
        return ec1, ec2

    def _calculate_vectors_west(self):
        ew1 = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )
        ew2 = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )
        ew1.data[:] = self._np.nan
        ew2.data[:] = self._np.nan
        ew1.data[1:-1, :-1, :3], ew2.data[1:-1, :-1, :3] = calc_unit_vector_west(
            self._dgrid_xyz,
            self._agrid_xyz,
            self._grid_type,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )
        return ew1, ew2

    def _calculate_vectors_south(self):
        es1 = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, CARTESIAN_DIM], ""
        )
        es2 = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, CARTESIAN_DIM], ""
        )
        es1.data[:] = self._np.nan
        es2.data[:] = self._np.nan
        es1.data[:-1, 1:-1, :3], es2.data[:-1, 1:-1, :3] = calc_unit_vector_south(
            self._dgrid_xyz,
            self._agrid_xyz,
            self._grid_type,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )
        return es1, es2

    def _calculate_more_trig_terms(self, cos_sg, sin_sg):
        cosa_u = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        cosa_v = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        cosa_s = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        sina_u = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        sina_v = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        rsin_u = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        rsin_v = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        rsina = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        rsin2 = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        cosa = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        sina = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        (
            cosa.data[:, :],
            sina.data[:, :],
            cosa_u.data[:, :-1],
            cosa_v.data[:-1, :],
            cosa_s.data[:-1, :-1],
            sina_u.data[:, :-1],
            sina_v.data[:-1, :],
            rsin_u.data[:, :-1],
            rsin_v.data[:-1, :],
            rsina.data[self._halo : -self._halo, self._halo : -self._halo],
            rsin2.data[:-1, :-1],
        ) = calculate_trig_uv(
            self._dgrid_xyz,
            cos_sg,
            sin_sg,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )
        return (
            cosa,
            sina,
            cosa_u,
            cosa_v,
            cosa_s,
            sina_u,
            sina_v,
            rsin_u,
            rsin_v,
            rsina,
            rsin2,
        )

    def _init_cell_trigonometry(self):

        self._cosa_u = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        self._cosa_v = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        self._cosa_s = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        self._sina_u = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        self._sina_v = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        self._rsin_u = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        self._rsin_v = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        self._rsina = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        self._rsin2 = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        self._cosa = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        self._sina = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], ""
        )

        cos_sg, sin_sg = calculate_supergrid_cos_sin(
            self._dgrid_xyz,
            self._agrid_xyz,
            self._ec1.data[:-1, :-1],
            self._ec2.data[:-1, :-1],
            self._grid_type,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )

        (
            self._cosa.data[:, :],
            self._sina.data[:, :],
            self._cosa_u.data[:, :-1],
            self._cosa_v.data[:-1, :],
            self._cosa_s.data[:-1, :-1],
            self._sina_u.data[:, :-1],
            self._sina_v.data[:-1, :],
            self._rsin_u.data[:, :-1],
            self._rsin_v.data[:-1, :],
            self._rsina.data[self._halo : -self._halo, self._halo : -self._halo],
            self._rsin2.data[:-1, :-1],
        ) = calculate_trig_uv(
            self._dgrid_xyz,
            cos_sg,
            sin_sg,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )

        supergrid_corner_fix(
            cos_sg, sin_sg, self._halo, self._tile_partitioner, self._rank
        )

        supergrid_trig = {}
        for i in range(1, 10):
            supergrid_trig[f"cos_sg{i}"] = self._quantity_factory.zeros(
                [fv3util.X_DIM, fv3util.Y_DIM], ""
            )
            supergrid_trig[f"cos_sg{i}"].data[:-1, :-1] = cos_sg[:, :, i - 1]
            supergrid_trig[f"sin_sg{i}"] = self._quantity_factory.zeros(
                [fv3util.X_DIM, fv3util.Y_DIM], ""
            )
            supergrid_trig[f"sin_sg{i}"].data[:-1, :-1] = sin_sg[:, :, i - 1]

        self._cos_sg1 = supergrid_trig["cos_sg1"]
        self._cos_sg2 = supergrid_trig["cos_sg2"]
        self._cos_sg3 = supergrid_trig["cos_sg3"]
        self._cos_sg4 = supergrid_trig["cos_sg4"]
        self._cos_sg5 = supergrid_trig["cos_sg5"]
        self._cos_sg6 = supergrid_trig["cos_sg6"]
        self._cos_sg7 = supergrid_trig["cos_sg7"]
        self._cos_sg8 = supergrid_trig["cos_sg8"]
        self._cos_sg9 = supergrid_trig["cos_sg9"]
        self._sin_sg1 = supergrid_trig["sin_sg1"]
        self._sin_sg2 = supergrid_trig["sin_sg2"]
        self._sin_sg3 = supergrid_trig["sin_sg3"]
        self._sin_sg4 = supergrid_trig["sin_sg4"]
        self._sin_sg5 = supergrid_trig["sin_sg5"]
        self._sin_sg6 = supergrid_trig["sin_sg6"]
        self._sin_sg7 = supergrid_trig["sin_sg7"]
        self._sin_sg8 = supergrid_trig["sin_sg8"]
        self._sin_sg9 = supergrid_trig["sin_sg9"]

    def _calculate_latlon_momentum_correction(self):
        l2c_v = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        l2c_u = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        (
            l2c_v.data[self._halo : -self._halo, self._halo : -self._halo - 1],
            l2c_u.data[self._halo : -self._halo - 1, self._halo : -self._halo],
        ) = calculate_l2c_vu(self._grid.data[:], self._halo, self._np)
        return l2c_v, l2c_u

    def _calculate_xy_unit_vectors(self):
        ee1 = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, CARTESIAN_DIM], ""
        )
        ee2 = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, CARTESIAN_DIM], ""
        )
        ee1.data[:] = self._np.nan
        ee2.data[:] = self._np.nan
        (
            ee1.data[self._halo : -self._halo, self._halo : -self._halo, :],
            ee2.data[self._halo : -self._halo, self._halo : -self._halo, :],
        ) = generate_xy_unit_vectors(
            self._dgrid_xyz, self._halo, self._tile_partitioner, self._rank, self._np
        )
        return ee1, ee2

    def _calculate_divg_del6(self):
        del6_u = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        del6_v = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        divg_u = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], ""
        )
        divg_v = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], ""
        )
        sin_sg = [
            self.sin_sg1.data[:-1, :-1],
            self.sin_sg2.data[:-1, :-1],
            self.sin_sg3.data[:-1, :-1],
            self.sin_sg4.data[:-1, :-1],
            self.sin_sg5.data[:-1, :-1],
        ]
        sin_sg = self._np.array(sin_sg).transpose(1, 2, 0)
        (
            divg_u.data[:-1, :],
            divg_v.data[:, :-1],
            del6_u.data[:-1, :],
            del6_v.data[:, :-1],
        ) = calculate_divg_del6(
            sin_sg,
            self.sina_u.data[:, :-1],
            self.sina_v.data[:-1, :],
            self.dx.data[:-1, :],
            self.dy.data[:, :-1],
            self.dxc.data[:, :-1],
            self.dyc.data[:-1, :],
            self._halo,
            self._tile_partitioner,
            self._rank,
        )
        self._comm.vector_halo_update(divg_v, divg_u, n_points=self._halo)
        self._comm.vector_halo_update(del6_v, del6_u, n_points=self._halo)
        # TODO: Add support for unsigned vector halo updates
        # instead of handling ad-hoc here
        divg_v.data[divg_v.data < 0] *= -1
        divg_u.data[divg_u.data < 0] *= -1
        del6_v.data[del6_v.data < 0] *= -1
        del6_u.data[del6_u.data < 0] *= -1
        return del6_u, del6_v, divg_u, divg_v

    def _calculate_unit_vectors_lonlat(self):
        vlon = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )
        vlat = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, CARTESIAN_DIM], ""
        )

        vlon.data[:-1, :-1], vlat.data[:-1, :-1] = unit_vector_lonlat(
            self._agrid.data[:-1, :-1], self._np
        )
        return vlon, vlat

    def _calculate_grid_z(self):
        z11 = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        z12 = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        z21 = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        z22 = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        (
            z11.data[:-1, :-1],
            z12.data[:-1, :-1],
            z21.data[:-1, :-1],
            z22.data[:-1, :-1],
        ) = calculate_grid_z(
            self.ec1.data[:-1, :-1],
            self.ec2.data[:-1, :-1],
            self.vlon.data[:-1, :-1],
            self.vlat.data[:-1, :-1],
            self._np,
        )
        return z11, z12, z21, z22

    def _calculate_grid_a(self):
        a11 = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        a12 = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        a21 = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        a22 = self._quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM], "")
        (
            a11.data[:-1, :-1],
            a12.data[:-1, :-1],
            a21.data[:-1, :-1],
            a22.data[:-1, :-1],
        ) = calculate_grid_a(
            self.z11.data[:-1, :-1],
            self.z12.data[:-1, :-1],
            self.z21.data[:-1, :-1],
            self.z22.data[:-1, :-1],
            self.sin_sg5.data[:-1, :-1],
        )
        return a11, a12, a21, a22

    def _calculate_edge_factors(self):
        nhalo = self._halo
        edge_s = self._quantity_factory.zeros([fv3util.X_INTERFACE_DIM], "")
        edge_n = self._quantity_factory.zeros([fv3util.X_INTERFACE_DIM], "")
        edge_e = self._quantity_factory.zeros([fv3util.Y_INTERFACE_DIM], "")
        edge_w = self._quantity_factory.zeros([fv3util.Y_INTERFACE_DIM], "")
        (
            edge_w.data[nhalo:-nhalo],
            edge_e.data[nhalo:-nhalo],
            edge_s.data[nhalo:-nhalo],
            edge_n.data[nhalo:-nhalo],
        ) = edge_factors(
            self.gridvar,
            self.agrid.data[:-1, :-1],
            self._grid_type,
            nhalo,
            self._tile_partitioner,
            self._rank,
            RADIUS,
            self._np,
        )
        return edge_w, edge_e, edge_s, edge_n

    def _calculate_edge_a2c_vect_factors(self):
        edge_vect_s = self._quantity_factory.zeros([fv3util.X_DIM], "")
        edge_vect_n = self._quantity_factory.zeros([fv3util.X_DIM], "")
        edge_vect_e = self._quantity_factory.zeros([fv3util.Y_DIM], "")
        edge_vect_w = self._quantity_factory.zeros([fv3util.Y_DIM], "")
        (
            edge_vect_w.data[:-1],
            edge_vect_e.data[:-1],
            edge_vect_s.data[:-1],
            edge_vect_n.data[:-1],
        ) = efactor_a2c_v(
            self.gridvar,
            self.agrid.data[:-1, :-1],
            self._grid_type,
            self._halo,
            self._tile_partitioner,
            self._rank,
            RADIUS,
            self._np,
        )
        return edge_vect_w, edge_vect_e, edge_vect_s, edge_vect_n

    def _reduce_global_area_minmaxes(self):
        min_area = self._np.min(self.area.data[:])
        max_area = self._np.max(self.area.data[:])
        min_area_c = self._np.min(self.area_c.data[:])
        max_area_c = self._np.max(self.area_c.data[:])

        try:
            self._da_min = self._comm.comm.allreduce(min_area, min)
            self._da_max = self._comm.comm.allreduce(max_area, max)
            self._da_min_c = self._comm.comm.allreduce(min_area_c, min)
            self._da_max_c = self._comm.comm.allreduce(max_area_c, max)
        except AttributeError:
            self._da_min = min_area
            self._da_max = max_area
            self._da_min_c = min_area_c
            self._da_max_c = max_area_c
