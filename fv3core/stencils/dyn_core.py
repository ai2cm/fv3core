from gt4py.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.stencils.c_sw as c_sw
import fv3core.stencils.d_sw as d_sw
import fv3core.stencils.del2cubed as del2cubed
import fv3core.stencils.nh_p_grad as nh_p_grad
import fv3core.stencils.pe_halo as pe_halo
import fv3core.stencils.pk3_halo as pk3_halo
import fv3core.stencils.ray_fast as ray_fast
import fv3core.stencils.riem_solver3 as riem_solver3
import fv3core.stencils.riem_solver_c as riem_solver_c
import fv3core.stencils.temperature_adjust as temperature_adjust
import fv3core.stencils.updatedzc as updatedzc
import fv3core.stencils.updatedzd as updatedzd
import fv3core.utils.global_config as global_config
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
import fv3gfs.util as fv3util
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK
from fv3core.decorators import FixedOriginStencil
import fv3gfs.util
from fv3core.utils.grid import axis_offsets
HUGE_R = 1.0e40


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
    from __externals__ import local_ie, local_is, local_je, local_js, hydrostatic

    with computation(PARALLEL), interval(...):
        if __INLINED(hydrostatic):
            wk = pkc[0, 0, 1] - pkc
        else:
            wk = delpc
        # TODO for PGradC validation only, not necessary for DynCore
        with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
            uc = uc + dt2 * rdxc / (wk[-1, 0, 0] + wk) * (
                (gz[-1, 0, 1] - gz) * (pkc[0, 0, 1] - pkc[-1, 0, 0])
                + (gz[-1, 0, 0] - gz[0, 0, 1]) * (pkc[-1, 0, 1] - pkc)
            )
        # TODO for PGradC validation only, not necessary for DynCore
        with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
            vc = vc + dt2 * rdyc / (wk[0, -1, 0] + wk) * (
                (gz[0, -1, 1] - gz) * (pkc[0, 0, 1] - pkc[0, -1, 0])
                + (gz[0, -1, 0] - gz[0, 0, 1]) * (pkc[0, -1, 1] - pkc)
            )


def get_n_con(namelist, grid):
    if namelist.convert_ke or namelist.vtdm4 > 1.0e-4:
        n_con = grid.npz
    else:
        if namelist.d2_bg_k1 < 1.0e-3:
            n_con = 0
        else:
            if namelist.d2_bg_k2 < 1.0e-3:
                n_con = 1
            else:
                n_con = 2
    return n_con


def dyncore_temporaries(shape, namelist, grid):
    tmps = {}
    utils.storage_dict(
        tmps,
        ["ut", "vt", "gz", "zh", "pem", "pkc", "pk3", "heat_source", "divgd"],
        shape,
        grid.full_origin(),
    )
    if not namelist.hydrostatic:
        # To write in parallel region, these need to be 3D first 
        utils.storage_dict(
            tmps,
            ["dp_ref", "zs"],
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

class AcousticDynamics:
    """
    Fortran name is dyn_core
    Peforms the Lagrangian acoustic dynamics described by Lin 2004
    """
    def __init__(self, comm: fv3gfs.util.CubedSphereCommunicator, namelist):
        self.comm = comm
        self.namelist = namelist
        self.grid = spec.grid
        self.do_halo_exchange = global_config.get_do_halo_exchange()
        self.n_con = get_n_con(namelist, self.grid)
        self.nonhydrostatic_pressure = nh_p_grad.NonHydrostaticPressureGradient()
        self._temporaries = dyncore_temporaries(self.grid.domain_shape_full(add=(1, 1, 1)), self.namelist, self.grid)
        self._dp_ref_compute = FixedOriginStencil(dp_ref_compute, origin=self.grid.full_origin(),domain=self.grid.domain_shape_full(add=(0, 0, 1)))
        self._set_gz = FixedOriginStencil(set_gz, origin=self.grid.compute_origin(), domain=self.grid.domain_shape_compute(add=(0, 0, 1)))
        self._set_pem = FixedOriginStencil(set_pem, origin=self.grid.compute_origin(add=(-1, -1, 0)), domain=self.grid.domain_shape_compute(add=(2, 2, 0)))
        pgradc_origin = self.grid.compute_origin()
        pgradc_domain= self.grid.domain_shape_compute(add=(1, 1, 0))
        ax_offsets = axis_offsets(self.grid, pgradc_origin, pgradc_domain)
        pgradc_kwargs = {"origin":  pgradc_origin, "domain": pgradc_domain, "externals":{"hydrostatic": self.namelist.hydrostatic, **ax_offsets}}
        self._p_grad_c = FixedOriginStencil(p_grad_c_stencil,  **pgradc_kwargs)
                            
    def __call__(self, state):
        # u, v, w, delz, delp, pt, pe, pk, phis, wsd, omga, ua, va, uc, vc, mfxd,
        # mfyd, cxd, cyd, pkz, peln, q_con, ak, bk, diss_estd, cappa, mdt, n_split,
        # akap, ptop, pfull, n_map, comm):
        grid = self.grid
        init_step = state.n_map == 1
        end_step = state.n_map == self.namelist.k_split
        akap = constants.KAPPA
        dt = state.mdt / self.namelist.n_split
        dt2 = 0.5 * dt
        hydrostatic = self.namelist.hydrostatic
        rgrav = 1.0 / constants.GRAV
        n_split = self.namelist.n_split
        # TODO: Put defaults into code.
        # m_split = 1. + abs(dt_atmos)/real(k_split*n_split*abs(p_split))
        # n_split = nint( real(n0split)/real(k_split*abs(p_split)) * stretch_fac + 0.5 )
        ms = max(1, self.namelist.m_split / 2.0)
        shape = state.delz.shape
        # NOTE: In Fortran model the halo update starts happens in fv_dynamics, not here.
        reqs = {}
        if self.do_halo_exchange:
            for halovar in [
                "q_con_quantity",
                "cappa_quantity",
                "delp_quantity",
                "pt_quantity",
            ]:
                reqs[halovar] = self.comm.start_halo_update(
                    state.__getattribute__(halovar), n_points=utils.halo
                )
            reqs_vector = self.comm.start_vector_halo_update(
                state.u_quantity, state.v_quantity, n_points=utils.halo
            )
            reqs["q_con_quantity"].wait()
            reqs["cappa_quantity"].wait()

        state.__dict__.update(self._temporaries)
        if init_step:
            state.gz[:-1, :-1, :] = HUGE_R
            state.diss_estd[grid.slice_dict(grid.compute_dict())] = 0.0
            if not hydrostatic:
                state.pk3[:-1, :-1, :] = HUGE_R
        state.mfxd[grid.slice_dict(grid.x3d_compute_dict())] = 0.0
        state.mfyd[grid.slice_dict(grid.y3d_compute_dict())] = 0.0
        state.cxd[grid.slice_dict(grid.x3d_compute_domain_y_dict())] = 0.0
        state.cyd[grid.slice_dict(grid.y3d_compute_domain_x_dict())] = 0.0

        if not hydrostatic:
            # k1k = akap / (1.0 - akap)
            self._dp_ref_compute(
                state.ak,
                state.bk,
                state.phis,
                state.dp_ref,
                state.zs,
                rgrav,
            )
            # After writing, make 'dp_ref' a K-field and 'zs' an IJ-field
            state.dp_ref = utils.make_storage_data(state.dp_ref[0, 0, :], (shape[2],), (0,))
            state.zs = utils.make_storage_data(state.zs[:, :, 0], shape[0:2], (0, 0))

        # "acoustic" loop
        # called this because its timestep is usually limited by horizontal sound-wave
        # processes. Note this is often not the limiting factor near the poles, where
        # the speed of the polar night jets can exceed two-thirds of the speed of sound.
        for it in range(n_split):
            # the Lagrangian dynamics have two parts. First we advance the C-grid winds
            # by half a time step (c_sw). Then the C-grid winds are used to define advective
            # fluxes to advance the D-grid prognostic fields a full time step
            # (the rest of the routines).
            #
            # Along-surface flux terms (mass, heat, vertical momentum, vorticity,
            # kinetic energy gradient terms) are evaluated forward-in-time.
            #
            # The pressure gradient force and elastic terms are then evaluated
            # backwards-in-time, to improve stability.
            remap_step = False
            if self.namelist.breed_vortex_inline or (it == n_split - 1):
                remap_step = True
            if not hydrostatic:
                if self.do_halo_exchange:
                    reqs["w_quantity"] = self.comm.start_halo_update(
                        state.w_quantity, n_points=utils.halo
                    )
                if it == 0:
                    self._set_gz(
                        state.zs,
                        state.delz,
                        state.gz,
                    )
                    if self.do_halo_exchange:
                        reqs["gz_quantity"] = self.comm.start_halo_update(
                            state.gz_quantity, n_points=utils.halo
                        )
            if it == 0:
                if self.do_halo_exchange:
                    reqs["delp_quantity"].wait()
                    reqs["pt_quantity"].wait()

            if it == n_split - 1 and end_step:
                if self.namelist.use_old_omega:  # apparently True
                    self._set_pem(
                        state.delp,
                        state.pem,
                        state.ptop,
                    )
            if self.do_halo_exchange:
                reqs_vector.wait()
                if not hydrostatic:
                    reqs["w_quantity"].wait()

            # compute the c-grid winds at t + 1/2 timestep
            state.delpc, state.ptc = c_sw.compute(
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
                    state.divgd_quantity, n_points=utils.halo
                )
            if not hydrostatic:
                if it == 0:
                    if self.do_halo_exchange:
                        reqs["gz_quantity"].wait()
                    copy_stencil(
                        state.gz,
                        state.zh,
                        origin=grid.full_origin(),
                        domain=grid.domain_shape_full(add=(0, 0, 1)),
                    )
                else:
                    copy_stencil(
                        state.zh,
                        state.gz,
                        origin=grid.full_origin(),
                        domain=grid.domain_shape_full(add=(0, 0, 1)),
                    )
            if not hydrostatic:
                updatedzc.compute(
                    state.dp_ref, state.zs, state.ut, state.vt, state.gz, state.ws3, dt2
                )
                riem_solver_c.compute(
                    ms,
                    dt2,
                    akap,
                    state.cappa,
                    state.ptop,
                    state.phis,
                    state.omga,
                    state.ptc,
                    state.q_con,
                    state.delpc,
                    state.gz,
                    state.pkc,
                    state.ws3,
                )

            self._p_grad_c(
                grid.rdxc,
                grid.rdyc,
                state.uc,
                state.vc,
                state.delpc,
                state.pkc,
                state.gz,
                dt2,
            )
            if self.do_halo_exchange:
                reqc_vector = self.comm.start_vector_halo_update(
                    state.uc_quantity, state.vc_quantity, n_points=utils.halo
                )
                if self.namelist.nord > 0:
                    reqs["divgd_quantity"].wait()
                reqc_vector.wait()
            # use the computed c-grid winds to evolve the d-grid winds forward
            # by 1 timestep
            d_sw.compute(
                state.vt,
                state.delp,
                state.ptc,
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
                    self.comm.halo_update(state.__getattribute__(halovar), n_points=utils.halo)

            # Not used unless we implement other betas and alternatives to nh_p_grad
            # if self.namelist.d_ext > 0:
            #    raise 'Unimplemented namelist option d_ext > 0'
            # else:
            #    divg2 = utils.make_storage_from_shape(delz.shape, grid.compute_origin())

            if not hydrostatic:
                updatedzd.compute(
                    state.dp_ref,
                    state.zs,
                    state.zh,
                    state.crx,
                    state.cry,
                    state.xfx,
                    state.yfx,
                    state.wsd,
                    dt,
                )

                riem_solver3.compute(
                    remap_step,
                    dt,
                    akap,
                    state.cappa,
                    state.ptop,
                    state.zs,
                    state.w,
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
                    state.wsd,
                )

                if self.do_halo_exchange:
                    reqs["zh_quantity"] = self.comm.start_halo_update(
                        state.zh_quantity, n_points=utils.halo
                    )
                    if grid.npx == grid.npy:
                        reqs["pkc_quantity"] = self.comm.start_halo_update(
                            state.pkc_quantity, n_points=2
                        )
                    else:
                        reqs["pkc_quantity"] = self.comm.start_halo_update(
                            state.pkc_quantity, n_points=utils.halo
                        )
                if remap_step:
                    pe_halo.compute(state.pe, state.delp, state.ptop)
                if self.namelist.use_logp:
                    raise Exception("unimplemented namelist option use_logp=True")
                else:
                    pk3_halo.compute(state.pk3, state.delp, state.ptop, akap)
            if not hydrostatic:
                if self.do_halo_exchange:
                    reqs["zh_quantity"].wait()
                    if grid.npx != grid.npy:
                        reqs["pkc_quantity"].wait()
                basic.multiply_constant(
                    state.zh,
                    state.gz,
                    constants.GRAV,
                    origin=(grid.is_ - 2, grid.js - 2, 0),
                    domain=(grid.nic + 4, grid.njc + 4, grid.npz + 1),
                )
                if grid.npx == grid.npy and self.do_halo_exchange:
                    reqs["pkc_quantity"].wait()
                if self.namelist.beta != 0:
                    raise Exception(
                        "Unimplemented namelist option -- we only support beta=0"
                    )
                self.nonhydrostatic_pressure(
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
                # TODO: Pass through ks, or remove, inconsistent representation vs Fortran.
                ray_fast.compute(
                    state.u,
                    state.v,
                    state.w,
                    state.dp_ref,
                    state.pfull,
                    dt,
                    state.ptop,
                    state.ks,
                )

            if self.do_halo_exchange:
                if it != n_split - 1:
                    reqs_vector = self.comm.start_vector_halo_update(
                        state.u_quantity, state.v_quantity, n_points=utils.halo
                    )
                else:
                    if self.namelist.grid_type < 4:
                        self.comm.synchronize_vector_interfaces(
                            state.u_quantity, state.v_quantity
                        )

        if self.n_con != 0 and self.namelist.d_con > 1.0e-5:
            nf_ke = min(3, self.namelist.nord + 1)

            if self.do_halo_exchange:
                self.comm.halo_update(state.heat_source_quantity, n_points=utils.halo)
            cd = constants.CNST_0P20 * grid.da_min
            del2cubed.compute(state.heat_source, nf_ke, cd, grid.npz)
            if not hydrostatic:
                temperature_adjust.compute(
                    state.pt,
                    state.pkz,
                    state.heat_source,
                    state.delz,
                    state.delp,
                    state.cappa,
                    self.n_con,
                    dt,
                )
