from typing import Tuple
from fv3core.grid.utils import set_eta, get_center_vector
from fv3core.utils.grid import GridIndexing
from .gnomonic import (
    get_area,
    gnomonic_grid,
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
from .utils import set_eta

import fv3gfs.util as fv3util
from fv3core.utils.corners import fill_corners_2d, fill_corners_agrid, fill_corners_dgrid, fill_corners_cgrid
from fv3core.utils.global_constants import PI, RADIUS, LON_OR_LAT_DIM, TILE_DIM
from fv3gfs.util.constants import N_HALO_DEFAULT
# TODO
# use cached_property, just for properties that have a single output per function
# pass in quantity factory, remove most other arguments
# use the halo default, don't pass it n, probably won't work
# get sizer from factory
# can corners use sizer rather than gridIndexer
class MetricTerms:

    def __init__(self,  grid, *, grid_type: int, layout: Tuple[int, int], npx: int, npy: int, npz: int, communicator, backend: str, halo=N_HALO_DEFAULT):
       
        self._halo = halo
        self._comm = communicator
        self._backend = backend
        self._quantity_factory, sizer = self._make_quantity_factory(layout, npx, npy, npz)
        self.grid_indexer = GridIndexing.from_sizer_and_communicator(sizer, self._comm)
       
        self._grid = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM], "radians", dtype=float
        )
    
        self._agrid = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM], "radians", dtype=float
        )
        self.grid=grid
        self._np = self._grid.np
        self._dx = None
        self._dy = None
        self._dx_agrid = None
        self._dy_agrid = None
        self._dx_cgrid = None
        self._dy_cgrid = None
        self._area = None
        self._area_c = None
        self._xyz_dgrid = None
        self._xyz_agrid = None
        self._init_dgrid(npx, npy, npz, grid_type)
        self._init_agrid()
    
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
    def area(self):
        if self._area is None:
            self._area = self._compute_area()
        return self._area
    
    @property
    def area_c(self):
        if self._area_c is None:
            self._area_c = self._compute_area_c()
        return self._area_c

    @property
    def _dgrid_xyz(self):
        if self._xyz_dgrid is None:
            self._xyz_dgrid = lon_lat_to_xyz(
                self._grid.data[:, :, 0], self._grid.data[:, :, 1], self._np
            )
        return self._xyz_dgrid
    
    @property
    def _agrid_xyz(self):
        if self._xyz_agrid is None: 
            self._xyz_agrid = lon_lat_to_xyz(
                self._agrid.data[:-1, :-1, 0],
                self._agrid.data[:-1, :-1, 1],
                self._np, 
            )
        return self._xyz_agrid
    
    def _make_quantity_factory(self, layout: Tuple[int, int], npx: int, npy: int, npz: int):
        #print('making quantity factory', npx, npy, self._halo, layout)
        sizer =  fv3util.SubtileGridSizer.from_tile_params(
            nx_tile=npx - 1,
            ny_tile=npy - 1,
            nz=npz,
            n_halo=self._halo,
            extra_dim_lengths={
                LON_OR_LAT_DIM: 2,
                TILE_DIM: 6,
            },
            layout=layout,
        )
        quantity_factory = fv3util.QuantityFactory.from_backend(
            sizer, backend=self._backend
        )
        return quantity_factory, sizer
    
    def _init_dgrid(self, npx: int, npy: int, npz: int, grid_type: int):
        # TODO size npx, npy, not local dims

        global_quantity_factory, _ = self._make_quantity_factory((1,1), npx, npy, npz)
        grid_global = global_quantity_factory.zeros(
            [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                LON_OR_LAT_DIM,
                TILE_DIM, 
            ],
            "radians",
            dtype=float,
        )

        tile0_lon = global_quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
        )
        tile0_lat = global_quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
        )
        gnomonic_grid(
            grid_type,
            tile0_lon.view[:],
            tile0_lat.view[:], 
            self._np,
        )
    
        grid_global.view[:, :, 0, 0] = tile0_lon.view[:]
        grid_global.view[:, :, 1, 0] = tile0_lat.view[:]
        mirror_grid(
            grid_global.data,
            self._halo,
            npx,
            npy,
            self._np,
        )
        # Shift the corner away from Japan
        # This will result in the corner close to east coast of China
        # TODO if not config.do_schmidt and config.shift_fac > 1.0e-4
        shift_fac = 18
        grid_global.view[:, :, 0, :] -= PI / shift_fac
        tile0_lon = grid_global.data[:, :, 0, :]
        tile0_lon[tile0_lon < 0] += 2 * PI
        grid_global.data[self._np.abs(grid_global.data[:]) < 1e-10] = 0.0
        # TODO, mpi scatter grid_global and subset grid_global for rank
        tile_index = self._comm.partitioner.tile_index(self._comm.rank)
        #print('hmmmm',self._grid.data.shape, grid_global.data.shape, self.grid.global_is, self.grid.global_ie+1, self.grid.global_js, self.grid.global_je+1, self.grid.is_,self.grid.ie+1, self.grid.js,self.grid.je+1,self.grid.rank)
        self._grid.data[self.grid.is_:self.grid.ie+2, self.grid.js:self.grid.je +2, :] = grid_global.data[self.grid.global_is:self.grid.global_ie+2, self.grid.global_js:self.grid.global_je+2, :, tile_index]
        self._comm.halo_update(self._grid, n_points=self._halo)
        
        fill_corners_2d(
            self._grid.data, self.grid_indexer, gridtype="B", direction="x"
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
            self.grid_indexer,
            gridtype="A",
            direction="x",
        )
        fill_corners_2d(
            self._agrid.data[:, :, 1][:, :, None],
            self.grid_indexer,
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
            self.grid_indexer,
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
           dx_agrid_tmp[:, :, None], dy_agrid_tmp[:, :, None], self.grid_indexer, vector=False
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
            self._comm.tile.partitioner,
            self._comm.rank,
            self._np,
        )
        set_tile_border_dyc(
            self._dgrid_xyz[3:-3, 3:-3, :],
            self._agrid_xyz[3:-3, 3:-3, :],
            RADIUS,
            dy_cgrid.data[3:-4, 3:-3],
            self._comm.tile.partitioner,
            self._comm.rank,
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
            self.grid_indexer,
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
        #set_corner_area_to_triangle_area(
        #    lon=self._agrid.data[2:-3, 2:-3, 0],
        #    lat=self._agrid.data[2:-3, 2:-3, 1],
        #    area=area_cgrid.data[3:-3, 3:-3],
        #    radius=RADIUS,
        #    np=self._np,
        #)

        set_c_grid_tile_border_area(
           self._dgrid_xyz[2:-2, 2:-2, :],
           self._agrid_xyz[2:-2, 2:-2, :],
            RADIUS,
            area_cgrid.data[3:-3, 3:-3],
            self._comm.tile.partitioner,
            self._comm.rank,
            self._np,
        )
        self._comm.halo_update(area_cgrid, n_points=self._halo)

        fill_corners_2d(
            area_cgrid.data[:, :, None],
            self.grid_indexer,
            gridtype="B",
            direction="x",
        )
        return area_cgrid
    

       
     

        
