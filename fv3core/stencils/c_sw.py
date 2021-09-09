import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.stencils.d2a2c_vect import DGrid2AGrid2CGridVectors
from fv3core.utils import corners
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


def geoadjust_ut(
    ut: FloatField,
    dy: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        ut[0, 0, 0] = (
            dt2 * ut * dy * sin_sg3[-1, 0] if ut > 0 else dt2 * ut * dy * sin_sg1
        )


def geoadjust_vt(
    vt: FloatField,
    dx: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        vt[0, 0, 0] = (
            dt2 * vt * dx * sin_sg4[0, -1] if vt > 0 else dt2 * vt * dx * sin_sg2
        )


def absolute_vorticity(vort: FloatField, fC: FloatFieldIJ, rarea_c: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = fC + rarea_c * vort


@gtscript.function
def nonhydro_x_fluxes(delp: FloatField, pt: FloatField, w: FloatField, utc: FloatField):
    fx1 = delp[-1, 0, 0] if utc > 0.0 else delp
    fx = pt[-1, 0, 0] if utc > 0.0 else pt
    fx2 = w[-1, 0, 0] if utc > 0.0 else w
    fx1 = utc * fx1
    fx = fx1 * fx
    fx2 = fx1 * fx2
    return fx, fx1, fx2


@gtscript.function
def nonhydro_y_fluxes(delp: FloatField, pt: FloatField, w: FloatField, vtc: FloatField):
    fy1 = delp[0, -1, 0] if vtc > 0.0 else delp
    fy = pt[0, -1, 0] if vtc > 0.0 else pt
    fy2 = w[0, -1, 0] if vtc > 0.0 else w
    fy1 = vtc * fy1
    fy = fy1 * fy
    fy2 = fy1 * fy2
    return fy, fy1, fy2


def compute_nonhydro_x_fluxes(
    delp: FloatField,
    pt: FloatField,
    utc: FloatField,
    w: FloatField,
    fx: FloatField,
    fx1: FloatField,
    fx2: FloatField,
):

    with computation(PARALLEL), interval(...):
        # delp = corners.fill_corners_2cells_x(delp)
        # pt = corners.fill_corners_2cells_x(pt)
        # w = corners.fill_corners_2cells_x(w)
        fx, fx1, fx2 = nonhydro_x_fluxes(delp, pt, w, utc)
        # delp = corners.fill_corners_2cells_y(delp)
        # pt = corners.fill_corners_2cells_y(pt)
        # w = corners.fill_corners_2cells_y(w)


def transportdelp_update_vorticity_and_kineticenergy(
    delp: FloatField,
    pt: FloatField,
    utc: FloatField,
    vtc: FloatField,
    w: FloatField,
    rarea: FloatFieldIJ,
    delpc: FloatField,
    ptc: FloatField,
    wc: FloatField,
    ke: FloatField,
    vort: FloatField,
    ua: FloatField,
    va: FloatField,
    uc: FloatField,
    vc: FloatField,
    u: FloatField,
    v: FloatField,
    fx: FloatField,
    fx1: FloatField,
    fx2: FloatField,
):

    with computation(PARALLEL), interval(...):
        fy, fy1, fy2 = nonhydro_y_fluxes(delp, pt, w, vtc)
        delpc = delp + (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
        ptc = (pt * delp + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / delpc
        wc = (w * delp + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea) / delpc
    with computation(PARALLEL), interval(...):
        # update vorticity and kinetic energy
        ke = uc if ua > 0.0 else uc[1, 0, 0]
        vort = vc if va > 0.0 else vc[0, 1, 0]


def north_south_edge_vorticity1(
    u: FloatField,
    va: FloatField,
    vort: FloatField,
    sin_sg4: FloatFieldIJ,
    cos_sg4: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        vort = vort * sin_sg4 + u[0, 1, 0] * cos_sg4 if va <= 0.0 else vort


def north_south_edge_vorticity2(
    u: FloatField,
    va: FloatField,
    vort: FloatField,
    sin_sg2: FloatFieldIJ,
    cos_sg2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        vort = vort * sin_sg2 + u * cos_sg2 if va > 0.0 else vort


def east_west_edge_kineticenergy1(
    v: FloatField,
    ua: FloatField,
    ke: FloatField,
    sin_sg3: FloatFieldIJ,
    cos_sg3: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        ke = ke * sin_sg3 + v[1, 0, 0] * cos_sg3 if ua <= 0.0 else ke


def east_west_edge_kineticenergy2(
    v: FloatField,
    ua: FloatField,
    ke: FloatField,
    sin_sg1: FloatFieldIJ,
    cos_sg1: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        ke = ke * sin_sg1 + v * cos_sg1 if ua > 0.0 else ke


def final_kineticenergy(
    ua: FloatField, va: FloatField, vort: FloatField, ke: FloatField, dt2: float
):
    with computation(PARALLEL), interval(...):
        ke = 0.5 * dt2 * (ua * ke + va * vort)


def uf_main(
    u: FloatField,
    va: FloatField,
    dyc: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    cos_sg2: FloatFieldIJ,
    cos_sg4: FloatFieldIJ,
    uf: FloatField,
):
    with computation(PARALLEL), interval(...):
        uf = (
            (u - 0.25 * (va[0, -1, 0] + va) * (cos_sg4[0, -1] + cos_sg2))
            * dyc
            * 0.5
            * (sin_sg4[0, -1] + sin_sg2)
        )


def vf_main(
    v: FloatField,
    ua: FloatField,
    dxc: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    cos_sg1: FloatFieldIJ,
    cos_sg3: FloatFieldIJ,
    vf: FloatField,
):
    with computation(PARALLEL), interval(...):
        vf = (
            (v - 0.25 * (ua[-1, 0, 0] + ua) * (cos_sg3[-1, 0] + cos_sg1))
            * dxc
            * 0.5
            * (sin_sg3[-1, 0] + sin_sg1)
        )


def uf_y_edge(
    u: FloatField,
    uf: FloatField,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    dyc: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        uf = u * dyc * 0.5 * (sin_sg4[0, -1] + sin_sg2)


def vf_x_edge(
    v: FloatField,
    vf: FloatField,
    sin_sg1: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    dxc: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        vf = v * dxc * 0.5 * (sin_sg3[-1, 0] + sin_sg1)


def divergence_main(uf: FloatField, vf: FloatField, divg_d: FloatField):
    with computation(PARALLEL), interval(...):
        divg_d = vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf


def divergence_south_corner(vf: FloatField, divg_d: FloatField):
    with computation(PARALLEL), interval(...):
        divg_d -= vf[0, -1, 0]


def divergence_north_corner(vf: FloatField, divg_d: FloatField):
    with computation(PARALLEL), interval(...):
        divg_d += vf


def divergence_main_final(rarea_c: FloatFieldIJ, divg_d: FloatField):
    with computation(PARALLEL), interval(...):
        divg_d *= rarea_c


def update_vorticity(
    uc: FloatField,
    vc: FloatField,
    dxc: FloatFieldIJ,
    dyc: FloatFieldIJ,
    vort_c: FloatField,
    fy: FloatField,
):
    """Update vort_c.

    Args:
        uc: x-velocity on C-grid (input)
        vc: y-velocity on C-grid (input)
        dxc: grid spacing in x-dir (input)
        dyc: grid spacing in y-dir (input)
        vort_c: C-grid vorticity (output)
    """

    with computation(PARALLEL), interval(...):
        fx = dxc * uc
        fy = dyc * vc
    with computation(PARALLEL), interval(...):
        vort_c = fx[0, -1, 0] - fx - fy[-1, 0, 0] + fy


def vorticity_west_corner(
    fy: FloatField,
    vort_c: FloatField,
):
    with computation(PARALLEL), interval(...):
        vort_c += fy[-1, 0, 0]


def vorticity_east_corner(
    fy: FloatField,
    vort_c: FloatField,
):
    with computation(PARALLEL), interval(...):
        vort_c -= fy[0, 0, 0]


def update_x_velocity(
    vorticity: FloatField,
    ke: FloatField,
    velocity: FloatField,
    velocity_c: FloatField,
    cosa: FloatFieldIJ,
    sina: FloatFieldIJ,
    rdxc: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[0, 1, 0]
        velocity_c = velocity_c + tmp_flux * flux + rdxc * (ke[-1, 0, 0] - ke)


def correct_x_edge_velocity(
    vorticity: FloatField,
    ke: FloatField,
    velocity: FloatField,
    velocity_c: FloatField,
    rdxc: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * velocity
        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[0, 1, 0]
        velocity_c = velocity_c + tmp_flux * flux + rdxc * (ke[-1, 0, 0] - ke)


def update_y_velocity(
    vorticity: FloatField,
    ke: FloatField,
    velocity: FloatField,
    velocity_c: FloatField,
    cosa: FloatFieldIJ,
    sina: FloatFieldIJ,
    rdyc: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[1, 0, 0]
        velocity_c = velocity_c - tmp_flux * flux + rdyc * (ke[0, -1, 0] - ke)


def correct_y_edge_velocity(
    vorticity: FloatField,
    ke: FloatField,
    velocity: FloatField,
    velocity_c: FloatField,
    rdyc: FloatFieldIJ,
    dt2: float,
):
    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * velocity
        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[1, 0, 0]
        velocity_c = velocity_c - tmp_flux * flux + rdyc * (ke[0, -1, 0] - ke)


def initialize_delpc_ptc(delpc: FloatField, ptc: FloatField):
    with computation(PARALLEL), interval(...):
        delpc = 0.0
        ptc = 0.0


class CGridShallowWaterDynamics:
    """
    Fortran name is c_sw
    """

    def __init__(self, grid, namelist):
        self.grid = grid
        self.namelist = namelist
        self._dord4 = True

        self._D2A2CGrid_Vectors = DGrid2AGrid2CGridVectors(
            self.grid, self.namelist, self._dord4
        )
        grid_type = self.namelist.grid_type
        origin_halo1 = (self.grid.is_ - 1, self.grid.js - 1, 0)
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self.delpc = utils.make_storage_from_shape(shape)
        self.ptc = utils.make_storage_from_shape(shape)
        self._tmp_uf = utils.make_storage_from_shape(shape)
        self._tmp_vf = utils.make_storage_from_shape(shape)
        self._tmp_fy = utils.make_storage_from_shape(shape)
        self._tmp_fx = utils.make_storage_from_shape(shape)
        self._tmp_fx1 = utils.make_storage_from_shape(shape)
        self._tmp_fx2 = utils.make_storage_from_shape(shape)
        corner_domain = (1, 1, self.grid.npz)
        self._initialize_delpc_ptc = FrozenStencil(
            func=initialize_delpc_ptc,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(),
        )

        self._ke = utils.make_storage_from_shape(shape)
        self._vort = utils.make_storage_from_shape(shape)
        origin = self.grid.compute_origin()
        domain = self.grid.domain_shape_compute(add=(1, 1, 0))
        ax_offsets = axis_offsets(self.grid, origin, domain)

        if self.namelist.nord > 0:
            self._uf_main = FrozenStencil(
                uf_main,
                origin=self.grid.compute_origin(add=(-1, 0, 0)),
                domain=self.grid.domain_shape_compute(add=(2, 1, 0)),
            )
            self._vf_main = FrozenStencil(
                vf_main,
                origin=self.grid.compute_origin(add=(0, -1, 0)),
                domain=self.grid.domain_shape_compute(add=(1, 2, 0)),
            )
            if self.grid.south_edge:
                self._uf_south_edge = FrozenStencil(
                    uf_y_edge,
                    origin=(self.grid.is_ - 1, self.grid.js, origin[2]),
                    domain=(self.grid.nic + 2, 1, domain[2]),
                )
            if self.grid.north_edge:
                self._uf_north_edge = FrozenStencil(
                    uf_y_edge,
                    origin=(self.grid.is_ - 1, self.grid.je + 1, origin[2]),
                    domain=(self.grid.nic + 2, 1, domain[2]),
                )
            if self.grid.west_edge:
                self._vf_west_edge = FrozenStencil(
                    vf_x_edge,
                    origin=(self.grid.is_, self.grid.js - 1, origin[2]),
                    domain=(1, self.grid.njc + 2, domain[2]),
                )
            if self.grid.east_edge:
                self._vf_east_edge = FrozenStencil(
                    vf_x_edge,
                    origin=(self.grid.ie + 1, self.grid.js - 1, origin[2]),
                    domain=(1, self.grid.njc + 2, domain[2]),
                )
            self._divergence_main = FrozenStencil(
                divergence_main,
                origin=origin,
                domain=domain,
            )

            if self.grid.sw_corner:
                self._divergence_sw_corner = FrozenStencil(
                    divergence_south_corner,
                    origin=self.grid.compute_origin(),
                    domain=corner_domain,
                )
            if self.grid.se_corner:
                self._divergence_se_corner = FrozenStencil(
                    divergence_south_corner,
                    origin=(self.grid.ie + 1, self.grid.js, 0),
                    domain=corner_domain,
                )
            if self.grid.nw_corner:
                self._divergence_nw_corner = FrozenStencil(
                    divergence_north_corner,
                    origin=(self.grid.ie + 1, self.grid.je + 1, 0),
                    domain=corner_domain,
                )
            if self.grid.ne_corner:
                self._divergence_ne_corner = FrozenStencil(
                    divergence_north_corner,
                    origin=(self.grid.is_, self.grid.je + 1, 0),
                    domain=corner_domain,
                )
            self._divergence_main_final = FrozenStencil(
                divergence_main_final,
                origin=origin,
                domain=domain,
            )
        geo_origin = (self.grid.is_ - 1, self.grid.js - 1, 0)
        self._geoadjust_ut = FrozenStencil(
            func=geoadjust_ut,
            origin=geo_origin,
            domain=(self.grid.nic + 3, self.grid.njc + 2, self.grid.npz),
        )
        self._geoadjust_vt = FrozenStencil(
            func=geoadjust_vt,
            origin=geo_origin,
            domain=(self.grid.nic + 2, self.grid.njc + 3, self.grid.npz),
        )

        self._compute_nonhydro_x_fluxes = FrozenStencil(
            compute_nonhydro_x_fluxes,
            origin=geo_origin,
            domain=self.grid.domain_shape_compute(add=(3, 2, 0)),
        )

        self._transportdelp_updatevorticity_and_ke = FrozenStencil(
            transportdelp_update_vorticity_and_kineticenergy,
            externals={
                "grid_type": grid_type,
            },
            origin=geo_origin,
            domain=self.grid.domain_shape_compute(add=(2, 2, 0)),
        )

        edge_domain_y = (self.grid.nic + 2, 1, self.grid.npz)
        if self.grid.south_edge:
            self._vorticity_edge_south1 = FrozenStencil(
                north_south_edge_vorticity1,
                origin=(self.grid.is_ - 1, self.grid.js - 1, 0),
                domain=edge_domain_y,
            )
            self._vorticity_edge_south2 = FrozenStencil(
                north_south_edge_vorticity2,
                origin=(self.grid.is_ - 1, self.grid.js, 0),
                domain=edge_domain_y,
            )

        if self.grid.north_edge:
            self._vorticity_edge_north1 = FrozenStencil(
                north_south_edge_vorticity1,
                origin=(self.grid.is_ - 1, self.grid.je, 0),
                domain=edge_domain_y,
            )
            self._vorticity_edge_north2 = FrozenStencil(
                north_south_edge_vorticity2,
                origin=(self.grid.is_ - 1, self.grid.je + 1, 0),
                domain=edge_domain_y,
            )

        edge_domain_x = (1, self.grid.njc + 2, self.grid.npz)
        if self.grid.west_edge:
            self._kineticenergy_edge_west1 = FrozenStencil(
                east_west_edge_kineticenergy1,
                origin=(self.grid.is_ - 1, self.grid.js - 1, 0),
                domain=edge_domain_x,
            )
            self._kineticenergy_edge_west2 = FrozenStencil(
                east_west_edge_kineticenergy2,
                origin=(self.grid.is_, self.grid.js - 1, 0),
                domain=edge_domain_x,
            )

        if self.grid.east_edge:
            self._kineticenergy_edge_east1 = FrozenStencil(
                east_west_edge_kineticenergy1,
                origin=(self.grid.ie, self.grid.js - 1, 0),
                domain=edge_domain_x,
            )
            self._kineticenergy_edge_east2 = FrozenStencil(
                east_west_edge_kineticenergy2,
                origin=(self.grid.ie + 1, self.grid.js - 1, 0),
                domain=edge_domain_x,
            )
        self._final_ke = FrozenStencil(
            final_kineticenergy,
            origin=geo_origin,
            domain=self.grid.domain_shape_compute(add=(2, 2, 0)),
        )
        self._update_vorticity = FrozenStencil(
            update_vorticity,
            origin=origin,
            domain=domain,
        )

        if self.grid.sw_corner:
            self._sw_corner_vorticity = FrozenStencil(
                vorticity_west_corner,
                origin=(self.grid.is_, self.grid.js, 0),
                domain=corner_domain,
            )
        if self.grid.nw_corner:
            self._nw_corner_vorticity = FrozenStencil(
                vorticity_west_corner,
                origin=(self.grid.is_, self.grid.je + 1, 0),
                domain=corner_domain,
            )
        if self.grid.se_corner:
            self._se_corner_vorticity = FrozenStencil(
                vorticity_east_corner,
                origin=(self.grid.ie + 1, self.grid.js, 0),
                domain=corner_domain,
            )
        if self.grid.ne_corner:
            self._ne_corner_vorticity = FrozenStencil(
                vorticity_east_corner,
                origin=(self.grid.ie + 1, self.grid.je + 1, 0),
                domain=corner_domain,
            )
        self._absolute_vorticity = FrozenStencil(
            func=absolute_vorticity,
            origin=origin,
            domain=(self.grid.nic + 1, self.grid.njc + 1, self.grid.npz),
        )

        js = self.grid.js + 1 if self.grid.south_edge else self.grid.js
        je = self.grid.je if self.grid.north_edge else self.grid.je + 1
        self._update_y_velocity = FrozenStencil(
            func=update_y_velocity,
            origin=(self.grid.is_, js, 0),
            domain=(self.grid.nic, je - js + 1, self.grid.npz),
        )

        if self.grid.south_edge:
            self._update_south_velocity = FrozenStencil(
                correct_y_edge_velocity,
                origin=(self.grid.is_, self.grid.js, 0),
                domain=(self.grid.nic, 1, self.grid.npz),
            )
        if self.grid.north_edge:
            self._update_north_velocity = FrozenStencil(
                correct_y_edge_velocity,
                origin=(self.grid.is_, self.grid.je + 1, 0),
                domain=(self.grid.nic, 1, self.grid.npz),
            )

        is_ = self.grid.is_ + 1 if self.grid.west_edge else self.grid.is_
        ie = self.grid.ie if self.grid.east_edge else self.grid.ie + 1
        self._update_x_velocity = FrozenStencil(
            func=update_x_velocity,
            origin=(is_, self.grid.js, 0),
            domain=(ie - is_ + 1, self.grid.njc, self.grid.npz),
        )

        if self.grid.west_edge:
            self._update_west_velocity = FrozenStencil(
                correct_x_edge_velocity,
                origin=(self.grid.is_, self.grid.js, 0),
                domain=(1, self.grid.njc, self.grid.npz),
            )
        if self.grid.east_edge:
            self._update_east_velocity = FrozenStencil(
                correct_x_edge_velocity,
                origin=(self.grid.ie + 1, self.grid.js, 0),
                domain=(1, self.grid.njc, self.grid.npz),
            )

    @computepath_method
    def _vorticitytransport_cgrid(
        self,
        uc,
        vc,
        vort_c,
        ke_c,
        v,
        u,
        dt2: float,
    ):
        """Update the C-Grid x and y velocity fields.

        Args:
            uc: x-velocity on C-grid (input, output)
            vc: y-velocity on C-grid (input, output)
            vort_c: Vorticity on C-grid (input)
            ke_c: kinetic energy on C-grid (input)
            v: y-velocity on D-grid (input)
            u: x-velocity on D-grid (input)
            dt2: timestep (input)
        """
        self._update_y_velocity(
            vort_c,
            ke_c,
            u,
            vc,
            self.grid.cosa_v,
            self.grid.sina_v,
            self.grid.rdyc,
            dt2,
        )
        if self.grid.south_edge:
            self._update_south_velocity(
                vort_c,
                ke_c,
                u,
                vc,
                self.grid.rdyc,
                dt2,
            )
        if self.grid.north_edge:
            self._update_north_velocity(
                vort_c,
                ke_c,
                u,
                vc,
                self.grid.rdyc,
                dt2,
            )
        self._update_x_velocity(
            vort_c,
            ke_c,
            v,
            uc,
            self.grid.cosa_u,
            self.grid.sina_u,
            self.grid.rdxc,
            dt2,
        )
        if self.grid.west_edge:
            self._update_west_velocity(
                vort_c,
                ke_c,
                v,
                uc,
                self.grid.rdxc,
                dt2,
            )
        if self.grid.west_edge:
            self._update_east_velocity(
                vort_c,
                ke_c,
                v,
                uc,
                self.grid.rdxc,
                dt2,
            )

    @computepath_method
    def __call__(
        self,
        delp,
        pt,
        u,
        v,
        w,
        uc,
        vc,
        ua,
        va,
        ut,
        vt,
        divgd,
        omga,
        dt2: float,
    ):
        """
        C-grid shallow water routine.

        Advances C-grid winds by half a time step.
        Args:
            delp: D-grid vertical delta in pressure (in)
            pt: D-grid potential temperature (in)
            u: D-grid x-velocity (in)
            v: D-grid y-velocity (in)
            w: vertical velocity (in)
            uc: C-grid x-velocity (inout)
            vc: C-grid y-velocity (inout)
            ua: A-grid x-velocity (in)
            va: A-grid y-velocity (in)
            ut: u * dx (inout)
            vt: v * dy (inout)
            divgd: D-grid horizontal divergence (inout)
            omga: Vertical pressure velocity (inout)
            dt2: Half a model timestep in seconds (in)
        """
        self._initialize_delpc_ptc(
            self.delpc,
            self.ptc,
        )
        self._D2A2CGrid_Vectors(uc, vc, u, v, ua, va, ut, vt)
        if self.namelist.nord > 0:
            self._uf_main(
                u,
                va,
                self.grid.dyc,
                self.grid.sin_sg2,
                self.grid.sin_sg4,
                self.grid.cos_sg2,
                self.grid.cos_sg4,
                self._tmp_uf,
            )
            self._vf_main(
                v,
                ua,
                self.grid.dxc,
                self.grid.sin_sg1,
                self.grid.sin_sg3,
                self.grid.cos_sg1,
                self.grid.cos_sg3,
                self._tmp_vf,
            )
            if self.grid.south_edge:
                self._uf_south_edge(
                    u, self._tmp_uf, self.grid.sin_sg2, self.grid.sin_sg4, self.grid.dyc
                )
            if self.grid.north_edge:
                self._uf_north_edge(
                    u, self._tmp_uf, self.grid.sin_sg2, self.grid.sin_sg4, self.grid.dyc
                )
            if self.grid.west_edge:
                self._vf_west_edge(
                    v, self._tmp_vf, self.grid.sin_sg1, self.grid.sin_sg3, self.grid.dxc
                )
            if self.grid.east_edge:
                self._vf_east_edge(
                    v, self._tmp_vf, self.grid.sin_sg1, self.grid.sin_sg3, self.grid.dxc
                )
            self._divergence_main(self._tmp_uf, self._tmp_vf, divgd)
            if self.grid.sw_corner:
                self._divergence_sw_corner(self._tmp_vf, divgd)
            if self.grid.se_corner:
                self._divergence_se_corner(self._tmp_vf, divgd)
            if self.grid.nw_corner:
                self._divergence_nw_corner(self._tmp_vf, divgd)
            if self.grid.ne_corner:
                self._divergence_ne_corner(self._tmp_vf, divgd)
            self._divergence_main_final(self.grid.rarea_c, divgd)
        self._geoadjust_ut(
            ut,
            self.grid.dy,
            self.grid.sin_sg3,
            self.grid.sin_sg1,
            dt2,
        )
        self._geoadjust_vt(
            vt,
            self.grid.dx,
            self.grid.sin_sg4,
            self.grid.sin_sg2,
            dt2,
        )
        corners.fill2_4corners(delp, pt, "x", self.grid)
        corners.fill_4corners(w, "x", self.grid)
        self._compute_nonhydro_x_fluxes(
            delp,
            pt,
            ut,
            w,
            self._tmp_fx,
            self._tmp_fx1,
            self._tmp_fx2,
        )
        corners.fill2_4corners(delp, pt, "y", self.grid)
        corners.fill_4corners(w, "y", self.grid)
        self._transportdelp_updatevorticity_and_ke(
            delp,
            pt,
            ut,
            vt,
            w,
            self.grid.rarea,
            self.delpc,
            self.ptc,
            omga,
            self._ke,
            self._vort,
            ua,
            va,
            uc,
            vc,
            u,
            v,
            self._tmp_fx,
            self._tmp_fx1,
            self._tmp_fx2,
        )
        if self.grid.south_edge:
            self._vorticity_edge_south1(
                u, va, self._vort, self.grid.sin_sg4, self.grid.cos_sg4
            )
            self._vorticity_edge_south2(
                u, va, self._vort, self.grid.sin_sg2, self.grid.cos_sg2
            )
        if self.grid.north_edge:
            self._vorticity_edge_north1(
                u, va, self._vort, self.grid.sin_sg4, self.grid.cos_sg4
            )
            self._vorticity_edge_north2(
                u, va, self._vort, self.grid.sin_sg2, self.grid.cos_sg2
            )

        if self.grid.west_edge:
            self._kineticenergy_edge_west1(
                v, ua, self._ke, self.grid.sin_sg3, self.grid.cos_sg3
            )
            self._kineticenergy_edge_west2(
                v, ua, self._ke, self.grid.sin_sg1, self.grid.cos_sg1
            )

        if self.grid.east_edge:
            self._kineticenergy_edge_east1(
                v, ua, self._ke, self.grid.sin_sg3, self.grid.cos_sg3
            )
            self._kineticenergy_edge_east2(
                v, ua, self._ke, self.grid.sin_sg1, self.grid.cos_sg1
            )
        self._final_ke(ua, va, self._vort, self._ke, dt2)
        self._update_vorticity(
            uc, vc, self.grid.dxc, self.grid.dyc, self._vort, self._tmp_fy
        )
        if self.grid.sw_corner:
            self._sw_corner_vorticity(self._tmp_fy, self._vort)
        if self.grid.nw_corner:
            self._nw_corner_vorticity(self._tmp_fy, self._vort)
        if self.grid.se_corner:
            self._se_corner_vorticity(self._tmp_fy, self._vort)
        if self.grid.ne_corner:
            self._ne_corner_vorticity(self._tmp_fy, self._vort)
        self._absolute_vorticity(
            self._vort,
            self.grid.fC,
            self.grid.rarea_c,
        )
        self._vorticitytransport_cgrid(uc, vc, self._vort, self._ke, v, u, dt2)
        return self.delpc, self.ptc
