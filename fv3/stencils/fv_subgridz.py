#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.utils.global_constants as constants
from fv3.stencils.basic_operations import dim
import fv3.stencils.copy_stencil as cp 
sd = utils.sd
RK = constants.CP_AIR / constants.RDGAS + 1.
G2 = 0.5 * constants.GRAV
T1_MIN = 160.
T2_MIN = 165.
T2_MAX = 315.
T3_MAX = 325.
USTAR2 = 1.e-4
RI_MAX = 1.
RI_MIN = 0.25
G2 = 0.5 * constants.GRAV

@gtscript.function
def standard_cm(q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel):
    q_liq = q0_liquid + q0_rain
    q_sol = q0_ice + q0_snow + q0_graupel
    cpm = (1. - (q0_vapor + q_liq + q_sol)) * constants.CP_AIR + q0_vapor * constants.CP_VAP + q_liq * constants.C_LIQ + q_sol * constants.C_ICE
    cvm = (1. - (q0_vapor + q_liq + q_sol)) * constants.CV_AIR + q0_vapor * constants.CV_VAP + q_liq * constants.C_LIQ + q_sol * constants.C_ICE
    return cpm, cvm
@gtscript.function
def tvol(gz, u0, v0, w0):
    return  gz + 0.5*(u0**2 + v0**2 + w0**2)
@utils.stencil()
def init_tvm(den: sd, gz: sd, gzh: sd, ta: sd, t0: sd, peln: sd, delp: sd, delz: sd, pm: sd, u0: sd, v0: sd, ua: sd, va: sd,
             w0: sd, w: sd, hd: sd, te: sd, q0_vapor: sd, q0_liquid: sd, q0_rain: sd, q0_ice: sd, q0_snow: sd, q0_graupel: sd, cvm: sd, cpm: sd, xvir: float):
    with computation(PARALLEL), interval(...):
        t0 = ta
        tvm = t0 * (1. + xvir*q0_vapor)
        u0 = ua
        v0 = va
        pm = delp / (peln[0, 0, 1] - peln)
    with computation(BACKWARD):interval(...):
        # note only for nwat = 6
        cpm, cvm = standard_cm(q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel)
        den = - delp / (constants.GRAV * delz)
        w0 = w
        gz = gzh - G2 * delz
        tmp = tvol(gz, u0, v0, w0)
        hd = cpm * t0 + tmp
        te = cvm*t0+ tmp
        gzh = gzh[0, 0, 1] - constants.GRAV * delz

@utils.stencil()
def m_loop(ri: sd, ri_rf: sd, mc: sd, q_con: sd, u0: sd, v0: sd, w0: sd, hd: sd, delp: sd, gz: sd,  q0_vapor: sd, q0_liquid: sd, q0_rain: sd, q0_ice: sd, q0_snow: sd, q0_graupel: sd, t_min: float, t_max: float, ratio: float):
    with computation(PARALLEL):interval(...):
        q_con = q0_l + q0_i + q0_s + q0_r + q0_g
    with computation(BACKWARD),interval(1, None):
        tv1 = t0[0, 0, -1] * (1. + xvir * q0_v[0, 0, -1] - q_con)
        tv2 = t0 * (1. + xvir * q0_v - q_con)
        pt1 = tv1 / pkz[0, 0, -1]
        pt2 = tv2 / pkz
        ri = (gz[0, 0, -1] - gz) * (pt1 - pt2)/(0.5 * (pt1 + pt2)* ((u0[0, 0, -1] - u0)**2 + (v0[0, 0, -1] - v0)**2 + USTAR2))
        if tv1 > t_max and tv1 > tv2:
            ri = 0
        elif tv2 < t_min:
            ri = ri if ri < 0.1 else 0.1
        # Adjustment for K-H instability:
        # Compute equivalent mass flux: mc
        # Add moist 2-dz instability consideration:
        ri_ref = RI_MIN + (RI_MAX - RI_MIN) * dim(400.e2, pm) / 200.e2
        if RI_MAX < ri_ref:
            ri_ref = RI_MAX
    with computation(BACKWARD):
        with interval(1, 2):
            ri_ref = 4. * ri_rf
        with interval(2, 3):
            ri_ref = 2. * ri_ref
        with interval(3, 4):
            ri_ref = 1.5 * ri_ref
    with computation(BACKWARD),interval(1, None):
        max_ri_ratio = ri/ri_ref
        if max_ri_ratio < 0.:
            max_ri_ration = 0.
        mc = ratio * delp[0, 0, -1] * delp / (delp[0, 0, -1] + delp) * (1. - max_ri_ratio)**2
        
@utils.stencil()
def q0_adjust(ri_rf: sd, mc: sd, q0):
    with computation(BACKWARD):
        with interval(-1, None):
            tmp = 0.
    with computation(BACKWARD):
        with interval(1, None):
            if ri < ri_rf:
                q0 = q0 + tmp
                h0 = mc * (q0 - q0[0, 0, -1])
                tmp = h0 / delp[0, 0, -1]
                q0 = q0 - tmp
        with interval(0, 1):
            if ri[0, 0, 1] < ri_rf[0, 0, 1]:
                q0 = q0 + tmp
@utils.stencil()
def mix_up(ri: sd, ri_rf: sd, u0: sd, v0:sd, w0: sd, te: sd, qcon: sd, q0_liquid: sd, q0_ice: sd, q0_snow: sd, q0_rain: sd, q0_graupel: sd):
    with computation(BACKWARD), interval(0, -1):
        if ri < ri_rf:
            qcon = q0_liquid + q0_ice + q0_snow + q0_rain + q0_graupel
    with computation(BACKWARD):
        with interval(-1, None):
            tmpu = 0.
            tmpv = 0.
            tmpw = 0.
            tmpte = 0.
    with computation(BACKWARD), interval(1, None):
        if ri < ri_rf:
            u0 = u0 + tmpu
            v0 = v0 + tmpv
            te = te + tmpte
            w0 = w0 + tmpw
            h0u = mc * (u0 - u0[0, 0, -1])
            tmpu = h0u / delp
            h0v = mc * (v0 - v0[0, 0, -1])
            tmpv = h0v / delp
            h0te = mc * (te - te[0, 0, -1])
            tmpte = h0te / delp
            h0w = mc * (w0 - w0[0, 0, -1])
            tmpw = h0w / delp
            u0 = u0 - tmpu
            v0 = v0 - tmpv
            w0 = w0 - tmpw
            te = te - tmpte
@utils.stencil()
def double_adjust_cvm(cvm: sd, cpm:sd, q0_liquid: sd, q0_vapor: sd, q0_ice: sd, q0_snow: sd, q0_rain: sd, q0_graupel: sd):
    with computation(BACKWARD), interval(1, None):
        cpm, cvm = standard_cm(q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel)
        tv = tvol(gz, u0, v0, w0)
        t0 = (te - tv) / cvm
        hd = cpm * t0 + tv
@gtscript.function
def readjust_by_frac(a0, a, fra):
    a0 = a + (a0 - a) * fra
    return a0

@utils.stencil()
def fraction_adjust(t0: sd, ta:sd, u0, ua, v0, va, fra: float):
    with computation(PARALLEL), interval(...):
        t0 = readjust_by_frac(t0, ta, fra)
        u0 = readjust_by_frac(u0, ua, fra)
        v0 = readjust_by_frac(v0, va, fra)

@utils.stencil()
def fraction_adjust_tracer(q0: sd, q:sd, fra: float):
    with computation(PARALLEL), interval(...):
        q0 = readjust_by_frac(q0, q, fra)

@utils.stencil()
def finalize(u0: sd, v0: sd, w0: sd, t0: sd, ua: sd, va: sd, ta: sd, w: sd, u_dt: sd, v_dt: sd, rdt: float):
    with computation(PARALLEL), interval(...):
        u_dt = rdt * (u0 - ua)
        v_dt = rdt * (v0 - va)
        ta = t0
        ua = u0
        va = v0
        w = w0
        
def compute(qvapor, ta, pe, k_bot, zvir, dt, nwat):
    grid = spec.grid
    rd = 1. / dt
    if k_bot is not None:
        if k_bot < 3:
            return
    else:
        k_bot = grid.npz
    if k_bot < min(grid.npz, 24):
        t_max = T2_MAX
    else:
        t_max = T3_MAX
    if pe[grid.is_, grid.js, 0] < 2.:
        t_min = T1_MIN
    else:
        t_min = T2_MIN

    if nwat == 0:
        xvir = 0.
        rz = 0
    else:
        xvir = zvir
        rz = constants.RV_GAS - constants.RDGAS
    m = 3
    fra = dt / float(spec.namelist('tau'))
    if spec.namelist['hydrostatic']:
        raise Exception('Hydrostatic not supported for fv_subgridz')
    q0 = {}
    for tracername in fv3utils.tracer_variables:
        q0[tracername] = cp.copy(state.__get_attribute__(tracername))
    origin = grid.compute_origin()
    shape = state.delp.shape
    u0 = utils.make_storage_from_shape(shape, origin)
    v0 = utils.make_storage_from_shape(shape, origin)
    w0 = utils.make_storage_from_shape(shape, origin)
    ghz = utils.make_storage_from_shape(shape, origin)
    gz = utils.make_storage_from_shape(shape, origin)
    t0 = utils.make_storage_from_shape(shape, origin)
    pm = utils.make_storage_from_shape(shape, origin)
    hd = utils.make_storage_from_shape(shape, origin)
    te = utils.make_storage_from_shape(shape, origin)
    den = utils.make_storage_from_shape(shape, origin)
   
    cvm = utils.make_storage_from_shape(shape, origin)
    cpm = utils.make_storage_from_shape(shape, origin)
    qs = utils.make_storage_from_shape(shape, origin)
    kbot_domain = (grid.nic, grid, njc, k_bot)
    origin = grid.compute_origin()
    init_tvm(den, gz, gzh, ta, t0, peln, delp, delz, pm, u0, v0, ua, va,
             w0, w, hd, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, cvm, cpm, origin=origin, domain=kbot_domain)
    ri = utils.make_storage_from_shape(shape, origin)
    ri_rf = utils.make_storage_from_shape(shape, origin)
    mc = utils.make_storage_from_shape(shape, origin)
    ratios = {0:0.25, 1:0.5, 2: 0.999}
    for n in range(m):
        ratio = ratios[n]
        m_loop(ri, ri_rf, mc, q_con, u0, v0, w0, hd, delp, gz, q0_liquid, q0_vapor, q0_rain, q0_ice, q0_snow, q0_graupel,
               t_min, t_max, ratio, origin=origin, domain=kbot_domain)
        for tracername in fv3utils.tracer_variables:
            q0_adjust(ri, ri_rf, mc, q0[tracername], origin=origin, domain=kbot_domain)
        mix_up(ri, ri_rf, u0, v0:sd, w0, te, qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel, origin=origin, domain=kbot_domain)
        double_adjust_cvm(cvm, cpm:sd, q0_liquid, q0_vapor, q0_ice, q0_snow, q0_rain, q0_graupel, origin=origin, domain=kbot_domain)
    if fra < 1.:
        fraction_adjust(t0, ta, u0, ua, v0, va, fra, origin=origin, domain=kbot_domain)
        for tracername in fv3utils.tracer_variables:
            fraction_adjust_tracer(q0[tracername], tracers[tracername], fra,  origin=origin, domain=kbot_domain)
    for tracername in fv3utils.tracer_variables:
        cp.copy_stencil(q0[tracername], tracers[tracername], origin=origin, domain=kbot_domain)
    finalize(u0, v0, w0, t0, ua, va, ta, w, u_dt, v_dt, rdt, origin=origin, domain=kbot_domain)
