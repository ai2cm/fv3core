from typing import Tuple
from fv3core.utils.grid import GridIndexing
from .geometry import get_center_vector
from .eta import set_eta

from .gnomonic import (
    get_area,
    local_gnomonic_ed,
    great_circle_distance_along_axis,
    lon_lat_corner_to_cell_center,
    lon_lat_midpoint,
    lon_lat_to_xyz,
    set_c_grid_tile_border_area,
    set_corner_area_to_triangle_area,
    set_tile_border_dxc,
    set_tile_border_dyc,
)

from .mirror import mirror_grid, set_halo_nan

import fv3gfs.util as fv3util
from fv3core.utils.corners import fill_corners_2d, fill_corners_agrid, fill_corners_dgrid, fill_corners_cgrid
from fv3core.utils.global_constants import PI, RADIUS, LON_OR_LAT_DIM, TILE_DIM
from fv3gfs.util.constants import N_HALO_DEFAULT
import functools
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
# corners use sizer + partitioner rather than GridIndexer, requires fv3core clls to corners know what to do
class MetricTerms:

    def __init__(self,  *, quantity_factory: fv3util.QuantityFactory, communicator: fv3util.Communicator, grid_type: int = 0):
        assert(grid_type < 3)
        self._halo = N_HALO_DEFAULT
        self._comm = communicator
        self._partitioner = self._comm.partitioner
        self._tile_partitioner = self._partitioner.tile
        self._rank = self._comm.rank
        self._quantity_factory = quantity_factory
        self._grid_indexer = GridIndexing.from_sizer_and_communicator(self._quantity_factory._sizer, self._comm)
        self._grid_dims = [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM]
        self._grid = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM], "radians", dtype=float
        )
        npx, npy, ndims  = self._tile_partitioner.global_extent(self._grid)
        self._npx = npx
        self._npy = npy
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
        
        self._init_dgrid()
        self._init_agrid()

    @classmethod
    def from_tile_sizing(cls, npx: int, npy: int, npz: int, communicator: fv3util.Communicator, backend: str, grid_type: int = 0) -> "MetricTerm":
        sizer =  fv3util.SubtileGridSizer.from_tile_params(
            nx_tile=npx - 1,
            ny_tile=npy - 1,
            nz=npz,
            n_halo=N_HALO_DEFAULT,
            extra_dim_lengths={
                LON_OR_LAT_DIM: 2,
                TILE_DIM: 6,
            },
            layout=communicator.partitioner.tile.layout,
        )
        quantity_factory = fv3util.QuantityFactory.from_backend(
            sizer, backend=backend
        )
        return cls(
            quantity_factory=quantity_factory,
            communicator=communicator,
            grid_type=grid_type
        )
 

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
    
    @cached_property
    def area(self):
        return  self._compute_area()
    
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
        return  lon_lat_to_xyz(
            self._agrid.data[:-1, :-1, 0],
            self._agrid.data[:-1, :-1, 1],
            self._np, 
        )
     
        
    def _init_dgrid(self):
     
        grid_mirror_ew = self._quantity_factory.zeros(self._grid_dims, "radians", dtype=float,)
        grid_mirror_ns = self._quantity_factory.zeros(self._grid_dims, "radians", dtype=float,)
        grid_mirror_diag = self._quantity_factory.zeros(self._grid_dims, "radians", dtype=float,)
      
        local_west_edge = self._tile_partitioner.on_tile_left(self._rank)
        local_east_edge = self._tile_partitioner.on_tile_right(self._rank)
        local_south_edge = self._tile_partitioner.on_tile_bottom(self._rank)
        local_north_edge = self._tile_partitioner.on_tile_top(self._rank)
        # information on position of subtile in full tile
        slice_x, slice_y = self._tile_partitioner.subtile_slice(self._rank, self._grid.dims, (self._npx, self._npy), overlap=True)
        section_global_is = self._halo + slice_x.start
        section_global_js = self._halo + slice_y.start
        subtile_width_x = slice_x.stop - slice_x.start - 1
        subtile_width_y = slice_y.stop - slice_y.start - 1

        # compute gnomonic grid for this rank
        local_gnomonic_ed( self._grid.view[:,:,0],
                           self._grid.view[:,:,1],
                           npx=self._npx,
                           west_edge=local_west_edge,
                           east_edge=local_east_edge,
                           south_edge=local_south_edge,
                           north_edge=local_north_edge,
                           global_is=section_global_is,
                           global_js=section_global_js,
                           np=self._np, rank=self._rank)
        
        # Next compute gnomonic for the mirrored ranks that'll be averaged
        j_subtile_index, i_subtile_index = self._tile_partitioner.subtile_index(self._rank)
        # compute the global index starting points for the mirrored ranks
        ew_global_is =  self._halo + (self._tile_partitioner.layout[0] - i_subtile_index - 1) * subtile_width_x
        ns_global_js =  self._halo +  (self._tile_partitioner.layout[1] - j_subtile_index - 1) * subtile_width_y
        
        # compute mirror in the east-west direction
        west_edge = True if local_east_edge else False
        east_edge = True if local_west_edge else False
        local_gnomonic_ed(grid_mirror_ew.view[:,:,0],
                          grid_mirror_ew.view[:,:,1],
                          npx=self._npx,
                          west_edge=west_edge,
                          east_edge=east_edge,
                          south_edge=local_south_edge,
                          north_edge=local_north_edge,
                          global_is=ew_global_is,
                          global_js=section_global_js,
                          np=self._np, rank=self._rank)

        # compute mirror in the north-south direction
        south_edge = True if local_north_edge else False
        north_edge = True if local_south_edge else False
        local_gnomonic_ed(grid_mirror_ns.view[:,:,0],
                          grid_mirror_ns.view[:,:,1],
                          npx=self._npx,
                          west_edge=local_west_edge,
                          east_edge=local_east_edge,
                          south_edge=south_edge,
                          north_edge=north_edge,
                          global_is=section_global_is,
                          global_js=ns_global_js,
                          np=self._np,
                          rank=self._rank)
           
        local_gnomonic_ed(grid_mirror_diag.view[:,:,0],
                          grid_mirror_diag.view[:,:,1],
                          npx=self._npx,
                          west_edge=west_edge,
                          east_edge=east_edge,
                          south_edge=south_edge,
                          north_edge=north_edge,
                          global_is=ew_global_is,
                          global_js=ns_global_js,
                          np=self._np,
                          rank=self._rank)
        
        # Average the mirrored gnomonic grids
        tile_index = self._partitioner.tile_index(self._rank) 
        mirror_data = {'local': self._grid.data, 'east-west': grid_mirror_ew.data, 'north-south': grid_mirror_ns.data, 'diagonal': grid_mirror_diag.data}
        mirror_grid(mirror_data=mirror_data,
                    tile_index=tile_index,
                    npx=self._npx,
                    npy=self._npy,
                    x_subtile_width=subtile_width_x + 1,
                    y_subtile_width=subtile_width_x + 1,
                    global_is=section_global_is,
                    global_js=section_global_js,
                    ng=self._halo,
                    np=self._grid.np,)
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
        #Set up lat-lon a-grid, calculate side lengths on a-grid
        lon_agrid, lat_agrid = lon_lat_corner_to_cell_center(self._grid.data[:, :, 0], self._grid.data[:, :, 1], self._np)
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
        dx = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], "m"
        )
       
        dx.view[:, :] = great_circle_distance_along_axis(
            self._grid.view[:, :, 0],
            self._grid.view[:, :, 1],
            RADIUS,
            self._np,
            axis=0,
        )
        dy = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], "m"
        )
        dy.view[:, :] = great_circle_distance_along_axis(
            self._grid.view[:, :, 0],
            self._grid.view[:, :, 1],
            RADIUS,
            self._np,
            axis=1,
        )
        self._comm.vector_halo_update(
            dx, dy, n_points=self._halo
        )

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
        return dx,dy

    
    def _compute_dxdy_agrid(self):
        
       dx_agrid = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m"
        )
       dy_agrid = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m"
        )
       lon, lat = self._grid.data[:, :, 0], self._grid.data[:, :, 1]
       lon_y_center, lat_y_center =  lon_lat_midpoint(
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
           dx_agrid_tmp[:, :, None], dy_agrid_tmp[:, :, None], self._grid_indexer, vector=False
       )
      
      
       dx_agrid.data[:-1, :-1] = dx_agrid_tmp
       dy_agrid.data[:-1, :-1] = dy_agrid_tmp
       self._comm.vector_halo_update(
           dx_agrid, dy_agrid, n_points=self._halo
       )
        
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

        lon_agrid, lat_agrid = self._agrid.data[:-1, :-1, 0],self._agrid.data[:-1, :-1, 1]
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
        self._comm.vector_halo_update(
            dx_cgrid, dy_cgrid, n_points=self._halo
        )

        #TODO: Add support for unsigned vector halo updates instead of handling ad-hoc here
        dx_cgrid.data[dx_cgrid.data < 0] *= -1
        dy_cgrid.data[dy_cgrid.data < 0] *= -1
        
        #TODO: fix issue with interface dimensions causing validation errors
        fill_corners_cgrid(
            dx_cgrid.data[:, :, None],
            dy_cgrid.data[:, :, None],
            self._grid_indexer,
            vector=False,
        )
       
        return dx_cgrid, dy_cgrid

    def _compute_area(self):
        area = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m^2"
        )
        area.data[:, :] = -1.e8
       
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
            rank = self._rank,
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
    

       
     

        
