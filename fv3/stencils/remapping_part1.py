#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.utils.global_constants as constants
import fv3.stencils.moist_cv as moist_cv
import fv3.stencils.saturation_adjustment as saturation_adjustment
import fv3.stencils.basic_operations as basic

# import fv3.stencils.map_scalar as map_scalar
import fv3.stencils.map_single as map_single

# import fv3.stencils.map_ppm_2d as map1_ppm
import fv3.stencils.mapn_tracer as mapn_tracer
import numpy as np
import fv3.stencils.copy_stencil as cp

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
    pe1: sd,
    pe2: sd,
    pe: sd,
    ak: sd,
    bk: sd,
    delp: sd,
    pe_bottom: sd,
    pn2: sd,
    peln: sd,
):
    with computation(FORWARD), interval(1, -1):
        pe2 = ak + bk * pe_bottom
    with computation(FORWARD), interval(0, -1):
        delp = pe2[0, 0, 1] - pe2
    # with computation(FORWARD), interval(...):
    #    pk1 = pk
    with computation(PARALLEL):
        with interval(0, 1):
            pn2 = peln
            # pk2 = pk
        with interval(-1, None):
            pn2 = peln
            # pk2 = pk


@utils.stencil()
def pressures_mapu(
    pe: sd, pe1: sd, ak: sd, bk: sd, pe_bottom: sd, pe1_bottom: sd, pe0: sd, pe3: sd
):
    with computation(FORWARD):
        with interval(0, 1):
            pe0 = pe
        with interval(1, None):
            pe0 = 0.5 * (pe[0, -1, 0] + pe1)
    with computation(FORWARD), interval(...):
        bkh = 0.5 * bk
        pe3 = ak + bkh * (pe_bottom[0, -1, 0] + pe1_bottom)


@utils.stencil()
def pressures_mapv(pe: sd, ak: sd, bk: sd, pe_bottom: sd, pe0: sd, pe3: sd):
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


# TODO remove this, this is a hack to deal with the fact that gz is a column
def reset_1d_x(gz):
    return utils.make_storage_data(np.squeeze(gz[:, :, spec.grid.npz - 1]), gz.shape)


def compute(
    qvapor,
    qliquid,
    qrain,
    qsnow,
    qice,
    qgraupel,
    qcld,
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
):
    grid = spec.grid
    hydrostatic = spec.namelist["hydrostatic"]
    nq = 7
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
                qvapor,
                qliquid,
                qice,
                qrain,
                qsnow,
                qgraupel,
                q_con,
                gz,
                cvm,
                pt,
                cappa,
                delp,
                delz,
                r_vir,
            )
            gz = reset_1d_x(gz)
            cvm = reset_1d_x(cvm)
    if not hydrostatic:
        delz_adjust(
            delp, delz, origin=grid.compute_origin(), domain=grid.domain_shape_compute()
        )
    pe_bottom = utils.make_storage_data(
        np.repeat(pe.data[:, :, grid.npz :], pe.shape[2], axis=2), pe.shape
    )
    pe1_bottom = utils.make_storage_data(
        np.repeat(pe1.data[:, :, grid.npz :], pe1.shape[2], axis=2), pe1.shape
    )
    # TODO ps is a 2d stencil...
    cp.copy_stencil(
        pe1_bottom,
        ps,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute_buffer_k(),
    )
    pressure_updates(
        pe1,
        pe2,
        pe,
        ak,
        bk,
        dp2,
        pe_bottom,
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
    # TODO merge into pressure updates when math available
    islice = slice(grid.is_, grid.ie + 1)
    jslice = slice(grid.js, grid.je + 1)
    kslice = slice(1, grid.npz)
    pn2[islice, jslice, kslice] = np.log(pe2[islice, jslice, kslice])
    pk[islice, jslice, kslice] = np.exp(akap * pn2[islice, jslice, kslice])
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
            184.0,
        )
    else:
        raise Exception("map ppm, untested mode where kord_tm >= 0")
        map_single.compute(
            pt, pe1, pe2, gz, 1, grid.is_, grid.ie, abs(spec.namelist["kord_tm"]), 184.0
        )
    # TODO if nq > 5:
    mapn_tracer.compute(
        pe1,
        pe2,
        dp2,
        qvapor,
        qliquid,
        qice,
        qrain,
        qsnow,
        qgraupel,
        qcld,
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
            qvapor,
            qliquid,
            qice,
            qrain,
            qsnow,
            qgraupel,
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
        # fix gz
        gz = reset_1d_x(gz)
        cvm = reset_1d_x(cvm)
    # if do_omega:
    # dp2 update, if larger than pe0 and smaller than one level up, update omega and  exit

    pe_bottom = utils.make_storage_data(
        np.repeat(pe.data[:, :, -1:], pe.shape[2], axis=2), pe.shape
    )
    pe1_bottom = utils.make_storage_data(
        np.repeat(pe1.data[:, :, -1:], pe.shape[2], axis=2), pe.shape
    )
    pressures_mapu(
        pe,
        pe1,
        ak,
        bk,
        pe_bottom,
        pe1_bottom,
        pe0,
        pe3,
        origin=grid.compute_origin(),
        domain=domain_jextra,
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
        pe,
        ak,
        bk,
        pe_bottom,
        pe0,
        pe3,
        origin=grid.compute_origin(),
        domain=domain_iextra,
    )
    map_single.compute(
        v, pe0, pe3, gz, -1, grid.is_, grid.ie + 1, spec.namelist["kord_mt"]
    )
    update_ua(pe2, ua, origin=grid.compute_origin(), domain=domain_jextra)
