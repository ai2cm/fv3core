#!/usr/bin/env python3
import fv3core.utils.gt4py_utils as utils
import fv3core._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3core.stencils.moist_cv as moist_cv
import fv3core.stencils.map_single as map_single
import fv3core.stencils.mapn_tracer as mapn_tracer
import fv3core.stencils.copy_stencil as cp

sd = utils.sd
CONSV_MIN = 0.001


@utils.stencil()
def init_pe2(pe: sd, pe2: sd, ptop: float):
    with computation(PARALLEL):
        with interval(0, 1):
            pe2 = ptop
        with interval(-1, None):
            pe2 = pe


@utils.stencil()
def delz_adjust(delp: sd, delz: sd):
    with computation(PARALLEL), interval(...):
        delz = -delz / delp


@utils.stencil()
def undo_delz_adjust(delp: sd, delz: sd):
    with computation(PARALLEL), interval(...):
        delz = -delz * delp


@utils.stencil()
def pressure_updates(
    pe1: sd, pe2: sd, pe: sd, ak: sd, bk: sd, delp: sd, ps: sd, pn2: sd, peln: sd,
):
    with computation(BACKWARD):
        with interval(-1, None):
            ps = pe
        with interval(0, -1):
            ps = ps[0, 0, 1]
    with computation(FORWARD), interval(1, -1):
        pe2 = ak + bk * ps
    with computation(FORWARD), interval(0, -1):
        delp = pe2[0, 0, 1] - pe2
    with computation(PARALLEL):
        with interval(0, 1):
            pn2 = peln
        with interval(-1, None):
            pn2 = peln


@utils.stencil()
def pn2_and_pk(pe2: sd, pn2: sd, pk: sd, akap: float):
    with computation(PARALLEL), interval(...):
        pn2 = log(pe2)
        pk = exp(akap * pn2)


@utils.stencil()
def pressures_mapu(pe: sd, pe1: sd, ak: sd, bk: sd, pe0: sd, pe3: sd):
    with computation(BACKWARD):
        with interval(-1, None):
            pe_bottom = pe
            pe1_bottom = pe
        with interval(0, -1):
            pe_bottom = pe_bottom[0, 0, 1]
            pe1_bottom = pe1_bottom[0, 0, 1]
    with computation(FORWARD):
        with interval(0, 1):
            pe0 = pe
        with interval(1, None):
            pe0 = 0.5 * (pe[0, -1, 0] + pe1)
    with computation(FORWARD), interval(...):
        bkh = 0.5 * bk
        pe3 = ak + bkh * (pe_bottom[0, -1, 0] + pe1_bottom)


@utils.stencil()
def pressures_mapv(pe: sd, ak: sd, bk: sd, pe0: sd, pe3: sd):
    with computation(BACKWARD):
        with interval(-1, None):
            pe_bottom = pe
        with interval(0, -1):
            pe_bottom = pe_bottom[0, 0, 1]
    with computation(FORWARD):
        with interval(0, 1):
            pe3 = ak
            pe0 = pe
        with interval(1, None):
            bkh = 0.5 * bk
            pe0 = 0.5 * (pe[-1, 0, 0] + pe)
            pe3 = ak + bkh * (pe_bottom[-1, 0, 0] + pe_bottom)


@utils.stencil()
def copy_j_adjacent(pe2: sd):
    with computation(PARALLEL), interval(...):
        pe2 = pe2[0, -1, 0]


@utils.stencil()
def update_ua(pe2: sd, ua: sd):
    with computation(PARALLEL), interval(0, -1):
        ua = pe2[0, 0, 1]


def compute(
    tracers,
    pt,
    delp,
    delz,
    peln,
    u,
    v,
    w,
    ua,
    cappa,
    q_con,
    pkz,
    pk,
    pe,
    hs,
    te,
    ps,
    wsd,
    omga,
    ak,
    bk,
    gz,
    cvm,
    ptop,
    akap,
    r_vir,
    nq,
):
    grid = spec.grid
    hydrostatic = spec.namelist["hydrostatic"]
    t_min = 184.0
    # do_omega = hydrostatic and last_step # TODO pull into inputs
    domain_jextra = (grid.nic, grid.njc + 1, grid.npz + 1)
    pe1 = cp.copy(pe, origin=grid.compute_origin(), domain=domain_jextra)
    pe2 = utils.make_storage_from_shape(pe.shape, grid.compute_origin())
    dp2 = utils.make_storage_from_shape(pe.shape, grid.compute_origin())
    pn2 = utils.make_storage_from_shape(pe.shape, grid.compute_origin())
    pe0 = utils.make_storage_from_shape(pe.shape, grid.compute_origin())
    pe3 = utils.make_storage_from_shape(pe.shape, grid.compute_origin())
    # pk2 = utils.make_storage_from_shape(pe.shape, grid.compute_origin())
    init_pe2(pe, pe2, ptop, origin=grid.compute_origin(), domain=domain_jextra)
    if spec.namelist["kord_tm"] < 0:
        if hydrostatic:
            raise Exception("Hydrostatic is not implemented")
        else:
            moist_cv.compute_pt(
                tracers["qvapor"],
                tracers["qliquid"],
                tracers["qice"],
                tracers["qrain"],
                tracers["qsnow"],
                tracers["qgraupel"],
                q_con,
                gz,
                cvm,
                pt,
                cappa,
                delp,
                delz,
                r_vir,
            )
    if not hydrostatic:
        delz_adjust(
            delp, delz, origin=grid.compute_origin(), domain=grid.domain_shape_compute()
        )
    pressure_updates(
        pe1,
        pe2,
        pe,
        ak,
        bk,
        dp2,
        ps,
        pn2,
        peln,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute_buffer_k(),
    )
    # TODO fix silly hack due to pe2 being 2d, so pe[:, je+1, 1:npz] should be the same as it was for pe[:, je, 1:npz] (unchanged)
    copy_j_adjacent(
        pe2, origin=(grid.is_, grid.je + 1, 1), domain=(grid.nic, 1, grid.npz - 1)
    )
    cp.copy_stencil(
        dp2, delp, origin=grid.compute_origin(), domain=grid.domain_shape_compute()
    )
    pn2_and_pk(
        pe2,
        pn2,
        pk,
        akap,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(),
    )
    if spec.namelist["kord_tm"] < 0:
        map_single.compute(
            pt,
            peln,
            pn2,
            gz,
            1,
            grid.is_,
            grid.ie,
            abs(spec.namelist["kord_tm"]),
            qmin=t_min,
        )
    else:
        raise Exception("map ppm, untested mode where kord_tm >= 0")
        map_single.compute(
            pt,
            pe1,
            pe2,
            gz,
            1,
            grid.is_,
            grid.ie,
            abs(spec.namelist["kord_tm"]),
            qmin=t_min,
        )
    # TODO if nq > 5:
    mapn_tracer.compute(
        pe1,
        pe2,
        dp2,
        tracers,
        nq,
        0.0,
        grid.is_,
        grid.ie,
        abs(spec.namelist["kord_tr"]),
    )
    # TODO else if nq > 0:
    # TODO map1_q2, fillz
    if not hydrostatic:
        map_single.compute(
            w, pe1, pe2, wsd, -2, grid.is_, grid.ie, spec.namelist["kord_wz"]
        )
        map_single.compute(
            delz, pe1, pe2, gz, 1, grid.is_, grid.ie, spec.namelist["kord_wz"]
        )
        undo_delz_adjust(
            delp, delz, origin=grid.compute_origin(), domain=grid.domain_shape_compute()
        )
    # if do_omega:  # NOTE untested
    #    pe3 = cp.copy(omga, (grid.is_, grid.js, 1))

    pe0 = cp.copy(
        peln, grid.compute_origin(), domain=(grid.nic, grid.njc, grid.npz + 1)
    )
    cp.copy_stencil(
        pn2,
        peln,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, grid.npz + 1),
    )
    if hydrostatic:
        # pkz
        pass
    else:
        moist_cv.compute_pkz(
            tracers["qvapor"],
            tracers["qliquid"],
            tracers["qice"],
            tracers["qrain"],
            tracers["qsnow"],
            tracers["qgraupel"],
            q_con,
            gz,
            cvm,
            pkz,
            pt,
            cappa,
            delp,
            delz,
            r_vir,
        )
    # if do_omega:
    # dp2 update, if larger than pe0 and smaller than one level up, update omega and  exit

    pressures_mapu(
        pe, pe1, ak, bk, pe0, pe3, origin=grid.compute_origin(), domain=domain_jextra,
    )
    map_single.compute(
        u,
        pe0,
        pe3,
        gz,
        -1,
        grid.is_,
        grid.ie,
        spec.namelist["kord_mt"],
        j_interface=True,
    )
    domain_iextra = (grid.nic + 1, grid.njc, grid.npz + 1)
    pressures_mapv(
        pe, ak, bk, pe0, pe3, origin=grid.compute_origin(), domain=domain_iextra,
    )
    map_single.compute(
        v, pe0, pe3, gz, -1, grid.is_, grid.ie + 1, spec.namelist["kord_mt"]
    )
    update_ua(pe2, ua, origin=grid.compute_origin(), domain=domain_jextra)
