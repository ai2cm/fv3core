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

class MetricTerms:

    def __init__(self,  grid_type, rank, layout, npx, npy, npz, halo, communicator, backend):
        self.npx = npx
        self.npy = npy
        self.npz = npz
        self.halo = halo
        self.rank = rank
        self.grid_type = grid_type
        self.layout = layout
        self._comm = communicator
        self._backend = backend
        self._quantity_factory, sizer = self._make_quantity_factory(layout)
        self.grid_indexer = GridIndexing.from_sizer_and_communicator(sizer, self._comm)
       
        self._grid = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM], "radians", dtype=float
        )
    
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
        self._area = None
        self._area_c = None
        self._xyz_dgrid = None
        self._xyz_agrid = None
        self._init_dgrid()
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
    
    def _make_quantity_factory(self, layout):
        sizer =  fv3util.SubtileGridSizer.from_tile_params(
            nx_tile=self.npx - 1,
            ny_tile=self.npy - 1,
            nz=self.npz,
            n_halo=self.halo,
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
    
    def _init_dgrid(self):
        # TODO size npx, npy, not local dims
        global_quantity_factory, _ = self._make_quantity_factory((1,1))
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
            self.grid_type,
            tile0_lon.view[:],
            tile0_lat.view[:], 
            self._np,
        )
       
        grid_global.view[:, :, 0, 0] = tile0_lon.view[:]
        grid_global.view[:, :, 1, 0] = tile0_lat.view[:]
        mirror_grid(
            grid_global.data,
            self.halo,
            self.npx,
            self.npy,
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
        self._grid.data[:] = grid_global.data[:, :, :, self.rank]
        self._comm.halo_update(self._grid, n_points=self.halo)
        
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
        self._comm.halo_update(self._agrid, n_points=self.halo)
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
            dx, dy, n_points=self.halo
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
           dx_agrid, dy_agrid, n_points=self.halo
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
            self._comm.tile.rank,
            self._np,
        )
        set_tile_border_dyc(
            self._dgrid_xyz[3:-3, 3:-3, :],
            self._agrid_xyz[3:-3, 3:-3, :],
            RADIUS,
            dy_cgrid.data[3:-4, 3:-3],
            self._comm.tile.partitioner,
            self._comm.tile.rank,
            self._np,
        )
        self._comm.vector_halo_update(
            dx_cgrid, dy_cgrid, n_points=self.halo
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
        self._comm.halo_update(area, n_points=self.halo)
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
        set_corner_area_to_triangle_area(
            lon=self._agrid.data[2:-3, 2:-3, 0],
            lat=self._agrid.data[2:-3, 2:-3, 1],
            area=area_cgrid.data[3:-3, 3:-3],
            radius=RADIUS,
            np=self._np,
        )
        set_c_grid_tile_border_area(
           self._dgrid_xyz[2:-2, 2:-2, :],
           self._agrid_xyz[2:-2, 2:-2, :],
            RADIUS,
            area_cgrid.data[3:-3, 3:-3],
            self._comm.tile.partitioner,
            self._comm.tile.rank,
            self._np,
        )
        self._comm.halo_update(area_cgrid, n_points=self.halo)
        fill_corners_2d(
            area_cgrid.data[:, :, None],
            self.grid_indexer,
            gridtype="B",
            direction="x",
        )
        return area_cgrid
    

       
     


class InitGrid:
    def __init__(self, grid_type, rank, layout, npx, npy, npz, halo, communicator, backend):
        self.npx = npx
        self.npy = npy
        self.npz = npz
        self.halo = halo
        self.rank = rank
        self.grid_type = grid_type
        self.layout = layout
        self._comm = communicator
        self._backend=backend
        self._sizer =  fv3util.SubtileGridSizer.from_tile_params(
            nx_tile=self.npx - 1,
            ny_tile=self.npy - 1,
            nz=self.npz,
            n_halo=self.halo,
            extra_dim_lengths={
                LON_OR_LAT_DIM: 2,
                TILE_DIM: 6,
            },
            layout=self.layout,
        )
        self._quantity_factory = fv3util.QuantityFactory.from_backend(
                self._sizer, backend=backend
            )
        self.grid_indexer = GridIndexing.from_sizer_and_communicator(self._sizer, self._comm)
       
    def generate(self):
        #Set up initial lat-lon d-grid
        shift_fac = 18
        global_sizer =  fv3util.SubtileGridSizer.from_tile_params(
            nx_tile=self.npx - 1,
            ny_tile=self.npy - 1,
            nz=self.npz,
            n_halo=self.halo,
            extra_dim_lengths={
                LON_OR_LAT_DIM: 2,
                TILE_DIM: 6,
            },
            layout=(1,1),
        )
        global_quantity_factory = fv3util.QuantityFactory.from_backend(
            global_sizer, backend=self._backend
        )
        grid_global = global_quantity_factory.zeros(
            [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                LON_OR_LAT_DIM,
                TILE_DIM, #TODO, only compute for this tile
            ],
            "radians",
            dtype=float,
        )
        state = {}
        # print(grid_global.np.shape(grid_global.data))
        # TODO )npx, npy, not local DIMS)
        tile0_lon = global_quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
        )
        tile0_lat = global_quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
        )
        gnomonic_grid(
            self.grid_type,
            tile0_lon.view[:],
            tile0_lat.view[:],
            tile0_lon.np,
        )

        grid_global.view[:, :, 0, 0] = tile0_lon.view[:]
        grid_global.view[:, :, 1, 0] = tile0_lat.view[:]
        mirror_grid(
            grid_global.data,
            self.halo,
            self.npx,
            self.npy,
            grid_global.np,
        )
        # Shift the corner away from Japan
        # This will result in the corner close to east coast of China
        grid_global.view[:, :, 0, :] -= PI / shift_fac
        
        tile0_lon = grid_global.data[:, :, 0, :]
        tile0_lon[tile0_lon < 0] += 2 * PI
        grid_global.data[grid_global.np.abs(grid_global.data[:]) < 1e-10] = 0.0
        state["lon"] = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
        )
        state["lat"] = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "radians", dtype=float
        )
        #state["lon"].data[:], state["lat"].data[:] = state["grid"].data[:, :, 0], state["grid"].data[:, :, 1]
        state["lon"].data[:], state["lat"].data[:] = grid_global.data[:, :, 0, self.rank], grid_global.data[:, :, 1, self.rank]
        state["grid"] = self._quantity_factory.empty(
            dims=[fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, LON_OR_LAT_DIM],
            units="radians",
        )
        #state["grid"].data[:] = grid_global.data[:, :, :, self.rank]
        #self._comm.halo_update(state["grid"], n_points=self.halo)
        #fill_corners_2d(
        #    state["grid"].data[:, :, :], self.grid_indexer, gridtype="B", direction="x"
        #)
        self._comm.halo_update(state["lon"], n_points=self.halo)
        self._comm.halo_update(state["lat"], n_points=self.halo)
        fill_corners_2d(
            state["lon"].data[:, :, None], self.grid_indexer, gridtype="B", direction="x"
        )
        fill_corners_2d(
            state["lat"].data[:, :, None], self.grid_indexer, gridtype="B", direction="x"
        )
        state["grid"].data[:,:,0] = state["lon"].data
        state["grid"].data[:,:,1] = state["lat"].data
        #calculate d-grid cell side lengths
        
        self._compute_local_dxdy(state)
        # before the halo update, the Fortran calls a get_symmetry routine
        # missing get_symmetry call in fv_grid_tools.F90, dy is set based on dx on
        # the opposite grid face, as a result dy has errors
        # (and dx in its halos from dy)
        self._comm.vector_halo_update(
                    state["dx"], state["dy"], n_points=self.halo
        )

        # at this point the Fortran code copies in the west and east edges from
        # the halo for dy and performs a halo update,
        # to ensure dx and dy mirror across the boundary.
        # Not doing it here at the moment.
        state["dx"].data[state["dx"].data < 0] *= -1
        state["dy"].data[state["dy"].data < 0] *= -1
        fill_corners_dgrid(
            state["dx"].data[:, :, None],
            state["dy"].data[:, :, None],
            self.grid_indexer,
            vector=False,
        )
        

        #Set up lat-lon a-grid, calculate side lengths on a-grid
        self._compute_local_agrid_part1(state)
        self._comm.halo_update(state["agrid"], n_points=self.halo)

        fill_corners_2d(
            state["agrid"].data[:, :, 0][:, :, None],
            self.grid_indexer,
            gridtype="A",
            direction="x",
        )
        fill_corners_2d(
            state["agrid"].data[:, :, 1][:, :, None],
            self.grid_indexer,
            gridtype="A",
            direction="y",
        )
        self._compute_local_agrid_part2(state)
        self._comm.vector_halo_update(
            state["dx_agrid"], state["dy_agrid"], n_points=self.halo
        )
        
        # at this point the Fortran code copies in the west and east edges from
        # the halo for dy and performs a halo update,
        # to ensure dx and dy mirror across the boundary.
        # Not doing it here at the moment.
        state["dx_agrid"].data[state["dx_agrid"].data < 0] *= -1
        state["dy_agrid"].data[state["dy_agrid"].data < 0] *= -1

        #Calculate a-grid areas and initial c-grid area
        self._compute_local_areas_pt1(state)
            

        #Finish c-grid areas, calculate sidelengths on the c-grid
        self._compute_local_areas_pt2(state, self._comm)
        self._comm.vector_halo_update(
            state["dx_cgrid"], state["dy_cgrid"], n_points=self.halo
        )

        #TODO: Add support for unsigned vector halo updates instead of handling ad-hoc here
        state["dx_cgrid"].data[state["dx_cgrid"].data < 0] *= -1
        state["dy_cgrid"].data[state["dy_cgrid"].data < 0] *= -1
        
        #TODO: fix issue with interface dimensions causing validation errors
        fill_corners_cgrid(
            state["dx_cgrid"].data[:, :, None],
            state["dy_cgrid"].data[:, :, None],
            self.grid_indexer,
            vector=False,
        )

        self._comm.halo_update(state["area"], n_points=self.halo)
        self._comm.halo_update(state["area_cgrid"], n_points=self.halo)
        fill_corners_2d(
            state["area_cgrid"].data[:, :, None][:, :, None],
            self.grid_indexer,
            gridtype="B",
            direction="x",
        )
        return state
    
    def _compute_local_dxdy(self, state):
        state["dx"] = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], "m"
        )
        state["dy"] = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], "m"
        )
        state["dx"].view[:, :] = great_circle_distance_along_axis(
            state["lon"].view[:],#[:, :, 0],
            state["lat"].view[:],#[:, :, 1],
            RADIUS,
            state["lon"].np,
            axis=0,
        )
        state["dy"].view[:, :] = great_circle_distance_along_axis(
            state["lon"].view[:],
            state["lat"].view[:],
            RADIUS,
            state["lon"].np,
            axis=1,
        )
        


    def _compute_local_agrid_part1(self, state):
        state["agrid"] = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM, LON_OR_LAT_DIM], "radians"
        )
        #lon, lat = state["grid"].data[:, :, 0], state["grid"].data[:, :, 1]
        agrid_lon, agrid_lat = lon_lat_corner_to_cell_center(state["lon"].data, state["lat"].data, state["grid"].np)
       
        state["agrid"].data[:-1, :-1, 0], state["agrid"].data[:-1, :-1, 1] = (
            agrid_lon,
            agrid_lat,
        )
        

    def _compute_local_agrid_part2(self, state):
        state["dx_agrid"] = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m"
        )
        state["dy_agrid"] = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m"
        )
        state["dx_cgrid"] = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM], "m"
        )
        state["dy_cgrid"] = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM], "m"
        )
        #lon, lat = state["grid"].data[:, :, 0], state["grid"].data[:, :, 1]
        lon_y_center, lat_y_center =  lon_lat_midpoint(
            state["lon"].data[:, :-1], state["lon"].data[:, 1:], state["lat"].data[:, :-1], state["lat"].data[:, 1:], state["grid"].np
        )
        dx_agrid = great_circle_distance_along_axis(
            lon_y_center, lat_y_center, RADIUS, state["grid"].np, axis=0
        )
        lon_x_center, lat_x_center = lon_lat_midpoint(
            state["lon"].data[:-1, :],  state["lon"].data[1:, :], state["lat"].data[:-1, :], state["lat"].data[1:, :], state["grid"].np
        )
        dy_agrid = great_circle_distance_along_axis(
            lon_x_center, lat_x_center, RADIUS, state["grid"].np, axis=1
        )
        fill_corners_agrid(
            dx_agrid[:, :, None], dy_agrid[:, :, None], self.grid_indexer, vector=False
        )
        lon_agrid, lat_agrid = (
            state["agrid"].data[:-1, :-1, 0],
            state["agrid"].data[:-1, :-1, 1],
        )
        dx_cgrid = great_circle_distance_along_axis(
            lon_agrid, lat_agrid, RADIUS, state["grid"].np, axis=0
        )
        dy_cgrid = great_circle_distance_along_axis(
            lon_agrid, lat_agrid, RADIUS, state["grid"].np, axis=1
        )

        state["dx_agrid"] = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m"
        )
        state["dy_agrid"] = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m"
        )
        state["dx_agrid"].data[:-1, :-1] = dx_agrid
        state["dy_agrid"].data[:-1, :-1] = dy_agrid

        # copying the second-to-last values to the last values is what the Fortran
        # code does, but is this correct/valid?
        # Maybe we want to change this to use halo updates?
        state["dx_cgrid"].data[1:-1, :-1] = dx_cgrid
        state["dx_cgrid"].data[0, :-1] = dx_cgrid[0, :]
        state["dx_cgrid"].data[-1, :-1] = dx_cgrid[-1, :]

        state["dy_cgrid"].data[:-1, 1:-1] = dy_cgrid
        state["dy_cgrid"].data[:-1, 0] = dy_cgrid[:, 0]
        state["dy_cgrid"].data[:-1, -1] = dy_cgrid[:, -1]

        


    def _compute_local_areas_pt1(self, state):
        state["area"] = self._quantity_factory.zeros(
            [fv3util.X_DIM, fv3util.Y_DIM], "m^2"
        )
        state["area"].data[:, :] = -1.e8
        state["area_cgrid"] = self._quantity_factory.zeros(
            [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM], "m^2"
        )
        state["area"].data[3:-4, 3:-4] = get_area(
            state["lon"].data[3:-3, 3:-3],
            state["lat"].data[3:-3, 3:-3],
            RADIUS,
            state["grid"].np,
        )
        state["area_cgrid"].data[3:-3, 3:-3] = get_area(
            state["agrid"].data[2:-3, 2:-3, 0],
            state["agrid"].data[2:-3, 2:-3, 1],
            RADIUS,
            state["grid"].np,
        )
        set_corner_area_to_triangle_area(
            lon=state["agrid"].data[2:-3, 2:-3, 0],
            lat=state["agrid"].data[2:-3, 2:-3, 1],
            area=state["area_cgrid"].data[3:-3, 3:-3],
            radius=RADIUS,
            np=state["grid"].np,
        )
        

    def _compute_local_areas_pt2(self, state, communicator):
        xyz_dgrid = lon_lat_to_xyz(
            state["lon"].data, state["lat"].data, state["grid"].np
        )
        xyz_agrid = lon_lat_to_xyz(
            state["agrid"].data[:-1, :-1, 0],
            state["agrid"].data[:-1, :-1, 1],
            state["agrid"].np,
        )
        set_c_grid_tile_border_area(
            xyz_dgrid[2:-2, 2:-2, :],
            xyz_agrid[2:-2, 2:-2, :],
            RADIUS,
            state["area_cgrid"].data[3:-3, 3:-3],
            communicator.tile.partitioner,
            self.rank,
            state["grid"].np,
        )
        set_tile_border_dxc(
            xyz_dgrid[3:-3, 3:-3, :],
            xyz_agrid[3:-3, 3:-3, :],
            RADIUS,
            state["dx_cgrid"].data[3:-3, 3:-4],
            communicator.tile.partitioner,
            self.rank,
            state["grid"].np,
        )
        set_tile_border_dyc(
            xyz_dgrid[3:-3, 3:-3, :],
            xyz_agrid[3:-3, 3:-3, :],
            RADIUS,
            state["dy_cgrid"].data[3:-4, 3:-3],
            communicator.tile.partitioner,
            self.rank,
            state["grid"].np,
        )
        
