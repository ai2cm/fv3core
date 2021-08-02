import dace
from gt4py.gtscript import __INLINED, BACKWARD, FORWARD, PARALLEL, computation, interval
import numpy as np

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.stencils.d_sw as d_sw
import fv3core.stencils.nh_p_grad as nh_p_grad
import fv3core.stencils.pe_halo as pe_halo
import fv3core.stencils.ray_fast as ray_fast
import fv3core.stencils.temperature_adjust as temperature_adjust
import fv3core.stencils.updatedzc as updatedzc
import fv3core.stencils.updatedzd as updatedzd
import fv3core.utils.global_config as global_config
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
import fv3gfs.util
import fv3gfs.util as fv3util
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.stencils.c_sw import CGridShallowWaterDynamics
from fv3core.stencils.del2cubed import HyperdiffusionDamping
from fv3core.stencils.pk3_halo import PK3Halo
from fv3core.stencils.riem_solver3 import RiemannSolver3
from fv3core.stencils.riem_solver_c import RiemannSolverC
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


HUGE_R = 1.0e40


def zero_data(
    mfxd: FloatField,
    mfyd: FloatField,
    cxd: FloatField,
    cyd: FloatField,
):
    with computation(PARALLEL), interval(...):
        mfxd = 0.0
        mfyd = 0.0
        cxd = 0.0
        cyd = 0.0


def zero_diss(
    diss_estd: FloatField,
    first_timestep: bool,
):
    with computation(PARALLEL), interval(...):
        if first_timestep:
            diss_estd = 0.0


# NOTE in Fortran these are columns
def dp_ref_compute(
    ak: FloatFieldK,
    bk: FloatFieldK,
    phis: FloatFieldIJ,
    dp_ref: FloatField,
    zs: FloatField,
    rgrav: float,
):
    with computation(PARALLEL), interval(0, -1):
        dp_ref = ak[1] - ak + (bk[1] - bk) * 1.0e5
    with computation(PARALLEL), interval(...):
        zs = phis * rgrav


def set_gz(zs: FloatFieldIJ, delz: FloatField, gz: FloatField):
    with computation(BACKWARD):
        with interval(-1, None):
            gz[0, 0, 0] = zs
        with interval(0, -1):
            gz[0, 0, 0] = gz[0, 0, 1] - delz


def set_pem(delp: FloatField, pem: FloatField, ptop: float):
    with computation(FORWARD):
        with interval(0, 1):
            pem[0, 0, 0] = ptop
        with interval(1, None):
            pem[0, 0, 0] = pem[0, 0, -1] + delp


def compute_geopotential(zh: FloatField, gz: FloatField):
    with computation(PARALLEL), interval(...):
        gz = zh * constants.GRAV


def p_grad_c_stencil(
    rdxc: FloatFieldIJ,
    rdyc: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    delpc: FloatField,
    pkc: FloatField,
    gz: FloatField,
    dt2: float,
):
    """Update C-grid winds from the pressure gradient force

    When this is run the C-grid winds have almost been completely
    updated by computing the momentum equation terms, but the pressure
    gradient force term has not yet been applied. This stencil completes
    the equation and Arakawa C-grid winds have been advected half a timestep
    upon completing this stencil..

     Args:
         uc: x-velocity on the C-grid (inout)
         vc: y-velocity on the C-grid (inout)
         delpc: vertical delta in pressure (in)
         pkc:  pressure if non-hydrostatic,
               (edge pressure)**(moist kappa) if hydrostatic(in)
         gz:  height of the model grid cells (m)(in)
         dt2: half a model timestep (for C-grid update) in seconds (in)
    Grid variable inputs:
        rdxc, rdyc
    """
    from __externals__ import hydrostatic

    with computation(PARALLEL), interval(...):
        if __INLINED(hydrostatic):
            wk = pkc[0, 0, 1] - pkc
        else:
            wk = delpc
        # TODO for PGradC validation only, not necessary for DynCore
        # with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
        uc = uc + dt2 * rdxc / (wk[-1, 0, 0] + wk) * (
            (gz[-1, 0, 1] - gz) * (pkc[0, 0, 1] - pkc[-1, 0, 0])
            + (gz[-1, 0, 0] - gz[0, 0, 1]) * (pkc[-1, 0, 1] - pkc)
        )
        # TODO for PGradC validation only, not necessary for DynCore
        # with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
        vc = vc + dt2 * rdyc / (wk[0, -1, 0] + wk) * (
            (gz[0, -1, 1] - gz) * (pkc[0, 0, 1] - pkc[0, -1, 0])
            + (gz[0, -1, 0] - gz[0, 0, 1]) * (pkc[0, -1, 1] - pkc)
        )


def get_nk_heat_dissipation(namelist, grid):
    # determines whether to convert dissipated kinetic energy into heat in the full
    # column, not at all, or in 1 or 2 of the top of atmosphere sponge layers
    if namelist.convert_ke or namelist.vtdm4 > 1.0e-4:
        nk_heat_dissipation = grid.npz
    else:
        if namelist.d2_bg_k1 < 1.0e-3:
            nk_heat_dissipation = 0
        else:
            if namelist.d2_bg_k2 < 1.0e-3:
                nk_heat_dissipation = 1
            else:
                nk_heat_dissipation = 2
    return nk_heat_dissipation


def dyncore_temporaries(shape, namelist, grid):
    tmps = {}
    utils.storage_dict(
        tmps,
        ["ut", "vt", "gz", "zh", "pem", "pkc", "pk3", "heat_source", "divgd"],
        shape,
        grid.full_origin(),
    )
    utils.storage_dict(
        tmps,
        ["ws3"],
        shape[0:2],
        grid.full_origin()[0:2],
    )
    utils.storage_dict(
        tmps, ["crx", "xfx"], shape, grid.compute_origin(add=(0, -grid.halo, 0))
    )
    utils.storage_dict(
        tmps, ["cry", "yfx"], shape, grid.compute_origin(add=(-grid.halo, 0, 0))
    )
    grid.quantity_dict_update(
        tmps, "heat_source", dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM]
    )
    for q in ["gz", "pkc", "zh"]:
        grid.quantity_dict_update(
            tmps, q, dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM]
        )
    grid.quantity_dict_update(
        tmps,
        "divgd",
        dims=[fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
    )

    return tmps


def _initialize_temp_adjust_stencil(grid, n_adj):
    """
    Returns the FrozenStencil Object for the temperature_adjust stencil
    Args:
        n_adj: Number of vertical levels to adjust temperature on
    """
    return FrozenStencil(
        temperature_adjust.compute_pkz_tempadjust,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, n_adj),
    )


class AcousticDynamics:
    """
    Fortran name is dyn_core
    Peforms the Lagrangian acoustic dynamics described by Lin 2004
    """

    # @computepath_method
    def dace_dummy(self, A):
        # self.__call__(state)
        return A + 2

    def __init__(
        self,
        comm: fv3gfs.util.CubedSphereCommunicator,
        namelist,
        ak: FloatFieldK,
        bk: FloatFieldK,
        pfull: FloatFieldK,
        phis: FloatFieldIJ,
    ):
        """
        Args:
            comm: object for cubed sphere inter-process communication
            namelist: flattened Fortran namelist
            ak: atmosphere hybrid a coordinate (Pa)
            bk: atmosphere hybrid b coordinate (dimensionless)
            phis: surface geopotential height
        """
        self.comm = comm
        self.namelist = namelist
        assert self.namelist.d_ext == 0, "d_ext != 0 is not implemented"
        assert self.namelist.beta == 0, "beta != 0 is not implemented"
        assert not self.namelist.use_logp, "use_logp=True is not implemented"
        self.grid = spec.grid
        self.do_halo_exchange = global_config.get_do_halo_exchange()
        self._pfull = pfull
        self._nk_heat_dissipation = get_nk_heat_dissipation(namelist, self.grid)
        self.nonhydrostatic_pressure_gradient = (
            nh_p_grad.NonHydrostaticPressureGradient(self.namelist.grid_type)
        )
        self._temporaries = dyncore_temporaries(
            self.grid.domain_shape_full(add=(1, 1, 1)), self.namelist, self.grid
        )
        self._temporaries["gz"][:] = HUGE_R
        if not namelist.hydrostatic:
            self._temporaries["pk3"][:] = HUGE_R

        column_namelist = d_sw.get_column_namelist(namelist, self.grid.npz)
        if not namelist.hydrostatic:
            # To write lower dimensional storages, these need to be 3D
            # then converted to lower dimensional
            dp_ref_3d = utils.make_storage_from_shape(
                self.grid.domain_shape_full(add=(1, 1, 1)), self.grid.full_origin()
            )
            zs_3d = utils.make_storage_from_shape(
                self.grid.domain_shape_full(add=(1, 1, 1)), self.grid.full_origin()
            )

            dp_ref_stencil = FrozenStencil(
                dp_ref_compute,
                origin=self.grid.full_origin(),
                domain=self.grid.domain_shape_full(add=(0, 0, 1)),
            )
            dp_ref_stencil(
                ak,
                bk,
                phis,
                dp_ref_3d,
                zs_3d,
                1.0 / constants.GRAV,
            )
            # After writing, make 'dp_ref' a K-field and 'zs' an IJ-field
            self._dp_ref = utils.make_storage_data(
                dp_ref_3d[0, 0, :], (dp_ref_3d.shape[2],), (0,)
            )
            self._zs = utils.make_storage_data(zs_3d[:, :, 0], zs_3d.shape[0:2], (0, 0))
            self.update_height_on_d_grid = updatedzd.UpdateHeightOnDGrid(
                self.grid, self.namelist, self._dp_ref, column_namelist, d_sw.k_bounds()
            )
            self.riem_solver3 = RiemannSolver3(namelist)
            self.riem_solver_c = RiemannSolverC(namelist)
            self._compute_geopotential_stencil = FrozenStencil(
                compute_geopotential,
                origin=(self.grid.is_ - 2, self.grid.js - 2, 0),
                domain=(self.grid.nic + 4, self.grid.njc + 4, self.grid.npz + 1),
            )
        self.dgrid_shallow_water_lagrangian_dynamics = (
            d_sw.DGridShallowWaterLagrangianDynamics(namelist, column_namelist)
        )
        self.cgrid_shallow_water_lagrangian_dynamics = CGridShallowWaterDynamics(
            self.grid, namelist
        )

        self._set_gz = FrozenStencil(
            set_gz,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(0, 0, 1)),
        )
        self._set_pem = FrozenStencil(
            set_pem,
            origin=self.grid.compute_origin(add=(-1, -1, 0)),
            domain=self.grid.domain_shape_compute(add=(2, 2, 0)),
        )

        pgradc_origin = self.grid.compute_origin()
        pgradc_domain = self.grid.domain_shape_compute(add=(1, 1, 0))
        self._p_grad_c = FrozenStencil(
            p_grad_c_stencil,
            origin=pgradc_origin,
            domain=pgradc_domain,
            externals={
                "hydrostatic": self.namelist.hydrostatic,
            },
        )

        self.update_geopotential_height_on_c_grid = (
            updatedzc.UpdateGeopotentialHeightOnCGrid(self.grid)
        )

        self._zero_data = FrozenStencil(
            zero_data,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(),
        )
        self._zero_diss = FrozenStencil(
            zero_diss,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        edge_domain_x = (1, self.grid.njc, self.grid.npz + 1)
        self._edge_pe_west_stencil = FrozenStencil(
            pe_halo.edge_pe,
            origin=(self.grid.is_ - 1, self.grid.js, 0),
            domain=edge_domain_x,
        )
        self._edge_pe_east_stencil = FrozenStencil(
            pe_halo.edge_pe,
            origin=(self.grid.ie + 1, self.grid.js, 0),
            domain=edge_domain_x,
        )
        edge_domain_y = (self.grid.nic + 2, 1, self.grid.npz + 1)
        self._edge_pe_south_stencil = FrozenStencil(
            pe_halo.edge_pe,
            origin=(self.grid.is_ - 1, self.grid.js - 1, 0),
            domain=edge_domain_y,
        )
        self._edge_pe_north_stencil = FrozenStencil(
            pe_halo.edge_pe,
            origin=(self.grid.is_ - 1, self.grid.je + 1, 0),
            domain=edge_domain_y,
        )
        """ The stencil object responsible for updading the interface pressure"""

        self._do_del2cubed = (
            self._nk_heat_dissipation != 0 and self.namelist.d_con > 1.0e-5
        )

        if self._do_del2cubed:
            nf_ke = min(3, self.namelist.nord + 1)
            self._hyperdiffusion = HyperdiffusionDamping(self.grid, nf_ke)
        if self.namelist.rf_fast:
            self._rayleigh_damping = ray_fast.RayleighDamping(self.grid, self.namelist)
        self._compute_pkz_tempadjust = _initialize_temp_adjust_stencil(
            self.grid,
            self._nk_heat_dissipation,
        )
        self._pk3_halo = PK3Halo(self.grid)
        self._copy_stencil = FrozenStencil(
            basic.copy_defn,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(0, 0, 1)),
        )

    @computepath_method(skip_dacemode=True)
    def __call__(self, state: dace.constant, insert_temporaries: dace.constant = True):
        # u, v, w, delz, delp, pt, pe, pk, phis, wsd, omga, ua, va, uc, vc, mfxd,
        # mfyd, cxd, cyd, pkz, peln, q_con, ak, bk, diss_estd, cappa, mdt, n_split,
        # akap, ptop, n_map, comm):
        if state.n_map == self.namelist.k_split:
            end_step = 1
        else:
            end_step = 0
        akap = constants.KAPPA
        dt = state.mdt / self.namelist.n_split
        dt2 = 0.5 * dt
        rgrav = 1.0 / constants.GRAV
        n_split = self.namelist.n_split
        # TODO: When the namelist values are set to 0, use these instead:
        # m_split = 1. + abs(dt_atmos)/real(k_split*n_split*abs(p_split))
        # n_split = nint( real(n0split)/real(k_split*abs(p_split)) * stretch_fac + 0.5 )
        ms = max(1, self.namelist.m_split / 2.0)
        # shape = state.delz.shape
        # NOTE: In Fortran model the halo update starts happens in fv_dynamics, not here
        if self.do_halo_exchange:
            reqs = {}
            for halovar in [
                "q_con_quantity",
                "cappa_quantity",
                "delp_quantity",
                "pt_quantity",
            ]:
                reqs[halovar] = self.comm.start_halo_update(
                    state.__getattribute__(halovar), n_points=self.grid.halo
                )
            reqs_vector = self.comm.start_vector_halo_update(
                state.u_quantity, state.v_quantity, n_points=self.grid.halo
            )
            reqs["q_con_quantity"].wait()
            reqs["cappa_quantity"].wait()

        if insert_temporaries:
            state.__dict__.update(self._temporaries)

        self._zero_data(
            state.mfxd,
            state.mfyd,
            state.cxd,
            state.cyd,
        )
        self._zero_diss(
            state.diss_estd,
            state.n_map == 1,
        )
        # "acoustic" loop
        # called this because its timestep is usually limited by horizontal sound-wave
        # processes. Note this is often not the limiting factor near the poles, where
        # the speed of the polar night jets can exceed two-thirds of the speed of sound.
        breed_vortex_inline = 1 if self.namelist.breed_vortex_inline else 0
        for it in range(n_split):
            # the Lagrangian dynamics have two parts. First we advance the C-grid winds
            # by half a time step (c_sw). Then the C-grid winds are used to define
            # advective fluxes to advance the D-grid prognostic fields a full time step
            # (the rest of the routines).
            #
            # Along-surface flux terms (mass, heat, vertical momentum, vorticity,
            # kinetic energy gradient terms) are evaluated forward-in-time.
            #
            # The pressure gradient force and elastic terms are then evaluated
            # backwards-in-time, to improve stability.
            remap_step = False
            if it == n_split - 1:
                tmpbool = True
            else:
                tmpbool = False
            if breed_vortex_inline or tmpbool:
                remap_step = True
            if not self.namelist.hydrostatic:
                if self.do_halo_exchange:
                    reqs["w_quantity"] = self.comm.start_halo_update(
                        state.w_quantity, n_points=self.grid.halo
                    )
                if it == 0:
                    self._set_gz(
                        self._zs,
                        state.delz,
                        state.gz,
                    )
                    if self.do_halo_exchange:
                        reqs["gz_quantity"] = self.comm.start_halo_update(
                            state.gz_quantity, n_points=self.grid.halo
                        )
            if it == 0:
                if self.do_halo_exchange:
                    reqs["delp_quantity"].wait()
                    reqs["pt_quantity"].wait()

            if it == n_split - 1 and end_step:
                if self.namelist.use_old_omega:
                    self._set_pem(
                        state.delp,
                        state.pem,
                        state.ptop,
                    )
            if self.do_halo_exchange:
                reqs_vector.wait()
                if not self.namelist.hydrostatic:
                    reqs["w_quantity"].wait()

            # compute the c-grid winds at t + 1/2 timestep
            delpc, ptc = self.cgrid_shallow_water_lagrangian_dynamics(
                state.delp,
                state.pt,
                state.u,
                state.v,
                state.w,
                state.uc,
                state.vc,
                state.ua,
                state.va,
                state.ut,
                state.vt,
                state.divgd,
                state.omga,
                dt2,
            )

            if self.namelist.nord > 0 and self.do_halo_exchange:
                reqs["divgd_quantity"] = self.comm.start_halo_update(
                    state.divgd_quantity, n_points=self.grid.halo
                )
            if not self.namelist.hydrostatic:
                if it == 0:
                    if self.do_halo_exchange:
                        reqs["gz_quantity"].wait()
                    self._copy_stencil(
                        state.gz,
                        state.zh,
                    )
                else:
                    self._copy_stencil(
                        state.zh,
                        state.gz,
                    )
            if not self.namelist.hydrostatic:
                self.update_geopotential_height_on_c_grid(
                    self._dp_ref, self._zs, state.ut, state.vt, state.gz, state.ws3, dt2
                )
                self.riem_solver_c(
                    dt2,
                    state.cappa,
                    state.ptop,
                    state.phis,
                    state.ws3,
                    ptc,
                    state.q_con,
                    delpc,
                    state.gz,
                    state.pkc,
                    state.omga,
                )

            self._p_grad_c(
                self.grid.rdxc,
                self.grid.rdyc,
                state.uc,
                state.vc,
                delpc,
                state.pkc,
                state.gz,
                dt2,
            )
            if self.do_halo_exchange:
                req_vector_c_grid = self.comm.start_vector_halo_update(
                    state.uc_quantity, state.vc_quantity, n_points=self.grid.halo
                )
                if self.namelist.nord > 0:
                    reqs["divgd_quantity"].wait()
                req_vector_c_grid.wait()
            # use the computed c-grid winds to evolve the d-grid winds forward
            # by 1 timestep
            self.dgrid_shallow_water_lagrangian_dynamics(
                state.vt,
                state.delp,
                ptc,
                state.pt,
                state.u,
                state.v,
                state.w,
                state.uc,
                state.vc,
                state.ua,
                state.va,
                state.divgd,
                state.mfxd,
                state.mfyd,
                state.cxd,
                state.cyd,
                state.crx,
                state.cry,
                state.xfx,
                state.yfx,
                state.q_con,
                state.zh,
                state.heat_source,
                state.diss_estd,
                dt,
            )
            # note that uc and vc are not needed at all past this point.
            # they will be re-computed from scratch on the next acoustic timestep.

            if self.do_halo_exchange:
                for halovar in ["delp_quantity", "pt_quantity", "q_con_quantity"]:
                    self.comm.halo_update(
                        state.__getattribute__(halovar), n_points=self.grid.halo
                    )

            # Not used unless we implement other betas and alternatives to nh_p_grad
            # if self.namelist.d_ext > 0:
            #    raise 'Unimplemented namelist option d_ext > 0'

            if not self.namelist.hydrostatic:
                self.update_height_on_d_grid(
                    self._zs,
                    state.zh,
                    state.crx,
                    state.cry,
                    state.xfx,
                    state.yfx,
                    state.wsd,
                    dt,
                )
                self.riem_solver3(
                    remap_step,
                    dt,
                    state.cappa,
                    state.ptop,
                    self._zs,
                    state.wsd,
                    state.delz,
                    state.q_con,
                    state.delp,
                    state.pt,
                    state.zh,
                    state.pe,
                    state.pkc,
                    state.pk3,
                    state.pk,
                    state.peln,
                    state.w,
                )

                if self.do_halo_exchange:
                    reqs["zh_quantity"] = self.comm.start_halo_update(
                        state.zh_quantity, n_points=self.grid.halo
                    )
                    if self.grid.npx == self.grid.npy:
                        reqs["pkc_quantity"] = self.comm.start_halo_update(
                            state.pkc_quantity, n_points=2
                        )
                    else:
                        reqs["pkc_quantity"] = self.comm.start_halo_update(
                            state.pkc_quantity, n_points=self.grid.halo
                        )
                if remap_step:
                    self._edge_pe_west_stencil(state.pe, state.delp, state.ptop)
                    self._edge_pe_east_stencil(state.pe, state.delp, state.ptop)
                    self._edge_pe_south_stencil(state.pe, state.delp, state.ptop)
                    self._edge_pe_north_stencil(state.pe, state.delp, state.ptop)
                if self.namelist.use_logp:
                    raise NotImplementedError(
                        "unimplemented namelist option use_logp=True"
                    )
                else:
                    self._pk3_halo(state.pk3, state.delp, state.ptop, akap)
            if not self.namelist.hydrostatic:
                if self.do_halo_exchange:
                    reqs["zh_quantity"].wait()
                    if self.grid.npx != self.grid.npy:
                        reqs["pkc_quantity"].wait()
                self._compute_geopotential_stencil(
                    state.zh,
                    state.gz,
                )
                if self.grid.npx == self.grid.npy and self.do_halo_exchange:
                    reqs["pkc_quantity"].wait()

                self.nonhydrostatic_pressure_gradient(
                    state.u,
                    state.v,
                    state.pkc,
                    state.gz,
                    state.pk3,
                    state.delp,
                    dt,
                    state.ptop,
                    akap,
                )

            if self.namelist.rf_fast:
                # TODO: Pass through ks, or remove, inconsistent representation vs
                # Fortran.
                self._rayleigh_damping(
                    state.u,
                    state.v,
                    state.w,
                    self._dp_ref,
                    self._pfull,
                    dt,
                    state.ptop,
                    state.ks,
                )

            if self.do_halo_exchange:
                if it != n_split - 1:
                    reqs_vector = self.comm.start_vector_halo_update(
                        state.u_quantity, state.v_quantity, n_points=self.grid.halo
                    )
                else:
                    if self.namelist.grid_type < 4:
                        self.comm.synchronize_vector_interfaces(
                            state.u_quantity, state.v_quantity
                        )

        if self._do_del2cubed:
            if self.do_halo_exchange:
                self.comm.halo_update(
                    state.heat_source_quantity, n_points=self.grid.halo
                )
            cd = constants.CNST_0P20 * self.grid.da_min
            self._hyperdiffusion(state.heat_source, cd)
            if not self.namelist.hydrostatic:
                tmp = dt * self.namelist.delt_max
                delt_time_factor = (tmp*tmp)**0.5
                self._compute_pkz_tempadjust(
                    state.delp,
                    state.delz,
                    state.cappa,
                    state.heat_source,
                    state.pt,
                    state.pkz,
                    delt_time_factor,
                )

