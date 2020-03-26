import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
import fv3._config as spec
import numpy as np
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd
origin = utils.origin

##
## Corner value stencils
##-----------------------

@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def compute_sw_corner(A: sd):
    with computation(PARALLEL), interval(...):
        A = (A + A[-1,0,0] + A[0,-1,0]) / 3.0

## 
## Flux value stencils
##---------------------
@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def compute_zonal_flux(flux: sd, A_in: sd, del_term: sd):
    with computation(PARALLEL), interval(...):
         flux = del_term * (A_in[-1,0,0] - A_in)

@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def compute_meridional_flux(flux: sd, A_in: sd, del_term: sd):
    with computation(PARALLEL), interval(...):
         flux = del_term * (A_in[0,-1,0] - A_in)

## 
## Q update stencil
##------------------
@gtscript.stencil(backend=utils.exec_backend, rebuild=True)
def update_q( q: sd, cd: float, rarea: sd, fx: sd, fy: sd ):
    with computation(PARALLEL), interval(...):
         q = q + cd*rarea*( fx - fx[1,0,0] + fy - fy[0,1,0] )

@gtscript.stencil(backend=utils.exec_backend)
def copy_row(A: sd):
    with computation(PARALLEL), interval(...):
       A = A[1,0,0] 

@gtscript.stencil(backend=utils.exec_backend)
def copy_column(A: sd):
    with computation(PARALLEL), interval(...):
       A = A[0,1,0] 

##
## Corner copy routine 
##---------------------
def copy_corner_values( grid, field, direction ):
    if grid.nested:
        return

    # x-coordinate direction
    if direction == 1:
        if grid.sw_corner:
            for j in range(grid.jsd,grid.js):
                for i in range(grid.isd,grid.is_):
                    ind = grid.js + grid.halo - j
                    field[ i,j,: ] = field[ j,ind,: ]
        if grid.se_corner:
            for j in range(grid.jsd,grid.js):
                for i in range(grid.ie+1,grid.ied+1):
                    ind = grid.je - grid.halo + j
                    ind_1 = grid.js + i - grid.ie
                    field[ i,j,: ] = field[ ind,ind_1,: ]
        if grid.ne_corner:
            for j in range(grid.je+1,grid.jed+1):
                for i in range(grid.ie+1,grid.ied+1):
                    ind = grid.je - (i-grid.ie-1)
                    field[ i,j,: ] = field[ j,ind,: ]
        if grid.nw_corner:
            for j in range(grid.je+1,grid.jed+1):
                for i in range(grid.isd,grid.is_):
                    ind = grid.is_ - (j-grid.je-1) 
                    ind_1 = grid.je - (i-grid.isd)
                    field[ i,j,: ] = field[ ind,ind_1,: ]

    # y-coordinate direction
    elif direction == 2:
        if grid.sw_corner:
            for j in range(grid.jsd,grid.js):
               for i in range(grid.isd,grid.is_):
                   i1 = grid.is_ + grid.halo - j
                   i2 = grid.jsd + i
                   field[ i,j,: ] = field[ i1,i2,: ]
        if grid.se_corner:
            for j in range(grid.jsd,grid.js):
                for i in range(grid.ie+1,grid.ied+1):
                    i1 = grid.ie - grid.halo + j
                    i2 = grid.js - 1 - (i-grid.ie-1)
                    field[ i,j,: ] = field[ i1,i2,: ]
        if grid.ne_corner:
            for j in range(grid.je+1,grid.jed+1):
                for i in range(grid.ie+1,grid.ied+1):
                    i1 = grid.ie - grid.halo - (j-grid.je)
                    i2 = grid.je + (i-grid.ie-1)
                    field[ i,j,: ] = field[ i1,i2,: ]
        if grid.nw_corner:
            for j in range(grid.je+1,grid.jed+1):
               for i in range(grid.isd,grid.is_):
                    i1 = grid.is_ + (j-grid.je-1)
                    i2 = grid.jsd - i
                    field[ i,j,: ] = field[ i1,i2,: ]

##
## corner_fill
##
## Subroutine that copies/fills in the appropriate corner values for qdel
##------------------------------------------------------------------------
def corner_fill( grid, q ):
    if grid.sw_corner:
       compute_sw_corner( q, origin=grid.compute_origin(), domain=grid.corner_domain() )
       copy_row( q, origin=(grid.is_-1,grid.js,0), domain=grid.corner_domain() )
       copy_column( q, origin=(grid.is_,grid.js-1,0), domain=grid.corner_domain() )
       
    if grid.se_corner:
       q[ grid.ie,grid.js,: ] = (q[ grid.ie,grid.js,: ] + q[ grid.npx,grid.js,: ] + q[ grid.ie,grid.js-1,: ])/3.0
       q[ grid.npx,grid.js,: ] = q[ grid.ie,grid.js,: ]
       copy_column( q, origin=(grid.ie,grid.js-1,0), domain=grid.corner_domain() )
       
    if grid.ne_corner:
       q[ grid.ie,grid.je,: ] = (q[ grid.ie,grid.je,: ] + q[ grid.npx,grid.je,: ] + q[ grid.ie,grid.je,: ])/3.0
       q[ grid.npx,grid.je,: ] = q[ grid.ie,grid.je,: ]
       q[ grid.ie,grid.npy,: ] = q[ grid.ie,grid.je,: ]
       
    if grid.nw_corner:
       q[ grid.is_,grid.je,: ] = (q[ grid.is_,grid.je,: ] + q[ grid.is_-1,grid.je,: ] + q[ grid.is_,grid.npy,: ])/3.0
       copy_row( q, origin=(grid.is_-1,grid.je,0), domain=grid.corner_domain() )
       q[ grid.is_,grid.npy,: ] = q[ grid.is_,grid.je,: ]

    return q

def compute(qdel, nmax, cd, km):
    grid = spec.grid
    origin = (grid.isd,grid.jsd,0)

    # Construct some necessary temporary storage objects
    fx = utils.make_storage_from_shape(qdel.shape, origin=origin)
    fy = utils.make_storage_from_shape(qdel.shape, origin=origin)

    # set up the temporal loop
    ntimes = min(3, nmax)
    for n in range(1,ntimes+1):
        nt = ntimes - n
        origin = (grid.is_-nt,grid.js-nt,0)

        # Fill in appropriate corner values
        qdel = corner_fill( grid, qdel )

#        if nt > 0:
#           copy_corner_values( grid, qdel, 1 )
        nx = (grid.ie+nt+1) - (grid.is_-nt) + 1
        ny = (grid.je+nt) - (grid.js-nt) + 1
        compute_zonal_flux( fx, qdel, grid.del6_v, origin=origin, domain=(nx,ny,km) )

#        if nt > 0:
#           copy_corner_values( grid, qdel, 2 )
        nx = (grid.ie+nt) - (grid.is_-nt) + 1
        ny = (grid.je+nt+1) - (grid.js-nt) + 1
        compute_meridional_flux( fy, qdel, grid.del6_u, origin=origin, domain=(nx,ny,km) )

        # Update q values
        ny = (grid.je+nt) - (grid.js-nt) + 1
        update_q( qdel, cd, grid.rarea, fx, fy, origin=origin, domain=(nx,ny,km))

