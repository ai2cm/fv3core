from .parallel_translate import ParallelTranslate


def init_grid(grid_spec_filename: str):
    """Reproduction of fv_grid_tools init_grid subroutine.

    Args:
        grid_spec_filename: only load the grid if this is INPUT/grid_spec.nc, otherwise
            ignore the file and generate a gnomonic grid

    """
    print(grid_spec_filename)
    raise NotImplementedError()


class TranslateInitGrid(ParallelTranslate):

    """
    !$ser
    data
    grid_file=grid_file
    ndims=ndims
    nregions=ntiles
    grid_name=grid_name
    sw_corner=Atm(n)%gridstruct%sw_corner
    se_corner=Atm(n)%gridstruct%se_corner
    ne_corner=Atm(n)%gridstruct%ne_corner
    nw_corner=Atm(n)%gridstruct%nw_corner
    """
    inputs = {
        "grid_file": {
            "name": "grid_spec_filename",
            "dims": [],
        }
    }
    """!$ser
data
iinta=Atm(n)%gridstruct%iinta
iintb=Atm(n)%gridstruct%iintb
jinta=Atm(n)%gridstruct%jinta
jintb=Atm(n)%gridstruct%jintb
gridvar=Atm(n)%gridstruct%grid_64
agrid=Atm(n)%gridstruct%agrid_64
area=Atm(n)%gridstruct%area_64
area_c=Atm(n)%gridstruct%area_c_64
rarea=Atm(n)%gridstruct%rarea
rarea_c=Atm(n)%gridstruct%rarea_c
dx=Atm(n)%gridstruct%dx_64
dy=Atm(n)%gridstruct%dy_64
dxc=Atm(n)%gridstruct%dxc_64
dyc=Atm(n)%gridstruct%dyc_64
dxa=Atm(n)%gridstruct%dxa_64
dya=Atm(n)%gridstruct%dya_64
rdx=Atm(n)%gridstruct%rdx
rdy=Atm(n)%gridstruct%rdy
rdxc=Atm(n)%gridstruct%rdxc
rdyc=Atm(n)%gridstruct%rdyc
rdxa=Atm(n)%gridstruct%rdxa
rdya=Atm(n)%gridstruct%rdya
latlon=Atm(n)%gridstruct%latlon
cubedsphere=Atm(n)%gridstruct%latlon
    """
    outputs = {}
    pass


class TranslateGridUtilsInit(ParallelTranslate):

    """!$ser
    data
    gridvar=Atm(n)%gridstruct%grid_64
    agrid=Atm(n)%gridstruct%agrid_64
    area=Atm(n)%gridstruct%area_64
    area_c=Atm(n)%gridstruct%area_c_64
    rarea=Atm(n)%gridstruct%rarea
    rarea_c=Atm(n)%gridstruct%rarea_c
    dx=Atm(n)%gridstruct%dx_64
    dy=Atm(n)%gridstruct%dy_64
    dxc=Atm(n)%gridstruct%dxc_64
    dyc=Atm(n)%gridstruct%dyc_64
    dxa=Atm(n)%gridstruct%dxa_64
    dya=Atm(n)%gridstruct%dya_64"""
    inputs = {}
    """!$ser
data
edge_s=Atm(n)%gridstruct%edge_s
edge_n=Atm(n)%gridstruct%edge_n
edge_w=Atm(n)%gridstruct%edge_w
edge_e=Atm(n)%gridstruct%edge_e
del6_u=Atm(n)%gridstruct%del6_u
del6_v=Atm(n)%gridstruct%del6_v
divg_u=Atm(n)%gridstruct%divg_u
divg_v=Atm(n)%gridstruct%divg_v
cosa_u=Atm(n)%gridstruct%cosa_u
cosa_v=Atm(n)%gridstruct%cosa_v
cosa_s=Atm(n)%gridstruct%cosa_s
cosa=Atm(n)%gridstruct%cosa
sina_u=Atm(n)%gridstruct%sina_u
sina_v=Atm(n)%gridstruct%sina_v
rsin_u=Atm(n)%gridstruct%rsin_u
rsin_v=Atm(n)%gridstruct%rsin_v
rsina=Atm(n)%gridstruct%rsina
rsin2=Atm(n)%gridstruct%rsin2
sina=Atm(n)%gridstruct%sina
sin_sg=Atm(n)%gridstruct%sin_sg
cos_sg=Atm(n)%gridstruct%cos_sg
ks=Atm(n)%ks
ptop=Atm(n)%ptop
ak=Atm(n)%ak
bk=Atm(n)%bk
a11=Atm(n)%gridstruct%a11
a12=Atm(n)%gridstruct%a12
a21=Atm(n)%gridstruct%a21
a22=Atm(n)%gridstruct%a22
da_min=Atm(n)%gridstruct%da_min
da_max=Atm(n)%gridstruct%da_max
da_min_c=Atm(n)%gridstruct%da_min_c
da_max_c=Atm(n)%gridstruct%da_max_c
sw_corner=Atm(n)%gridstruct%sw_corner
se_corner=Atm(n)%gridstruct%se_corner
ne_corner=Atm(n)%gridstruct%ne_corner
nw_corner=Atm(n)%gridstruct%nw_corner"""
    outputs = {}
