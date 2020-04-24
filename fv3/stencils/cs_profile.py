import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
import fv3.stencils.cs_limiters as cs_limiters
import numpy as np

sd = utils.sd


def grid():
    return spec.grid


@gtscript.function
def absolute_value(in_array):
    abs_value = in_array if in_array > 0 else -in_array
    return abs_value

@gtscript.function
def limit_minmax(q, a4):
    tmp = a4[0,0,-1] if a4[0,0,-1] > a4 else a4
    ret = q if q < tmp else tmp
    return ret

@gtscript.function
def limit_maxmin(q, a4):
    tmp2 = a4[0,0,-1] if a4[0,0,-1] < a4 else a4
    ret = q if q > tmp2 else tmp2
    return ret

@gtscript.function
def limit_both(q, a4):
    ret = limit_minmax(q,a4)
    ret = limit_maxmin(ret,a4)
    return ret

@gtscript.function
def constrain_interior(q, gam, a4):
    return limit_both(q, a4) if (gam[0,0,-1]*gam[0,0,1]>0.) else limit_maxmin(q, a4) if (gam[0,0,-1] > 0.) else limit_minmax(q, a4)

@utils.stencil()
def set_vals_2(gam:sd, q:sd, delp:sd, a4_1:sd, q_bot:sd, qs:sd):
    with computation(PARALLEL):
        with interval(0, 2):
            # set top
            gam = 0.5
            q = 1.5 * a4_1
    with computation(FORWARD):
        with interval(1, -1):
            # set middle
            grid_ratio = delp[0, 0, -1] / delp
            bet = 2.0 + grid_ratio + grid_ratio - gam
            q = (3.0 * (a4_1[0, 0, -1] + a4_1[0, 0, 0]) - q[0, 0, -1]) / bet
            gam[0, 0, +1] = grid_ratio / bet
    with computation(PARALLEL):
        with interval(-1,None):
            #set bottom
            grid_ratio = delp[0, 0, -1] / delp
            q = (3. * (a4_1[0,0,-1] + a4_1[0,0,0]) - (qs * grid_ratio) - q[0,0,-1]) / ( 2. + grid_ratio + grid_ratio - gam)
            q_bot = qs
    with computation(BACKWARD), interval(0,-1):
        q = q - (gam[0,0,1] * q[0,0,1])

@utils.stencil()
def set_vals_1(gam:sd, q:sd, delp:sd, a4_1:sd, q_bot:sd):
    with computation(PARALLEL):
        with interval(0, 1):
            # set top
            grid_ratio = delp[0, 0, 1] / delp
            bet = grid_ratio * (grid_ratio + 0.5)
            q = ((grid_ratio + grid_ratio) * (grid_ratio + 1.0) * a4_1 + a4_1[0, 0, 1]) / bet
            gam = (1.0 + grid_ratio * (grid_ratio + 1.5)) / bet
    with computation(FORWARD):
        with interval(1,-1):
            #set middle
            d4 = delp[0,0,-1] / delp
            bet = 2. + d4 + d4 - gam[0,0,-1]
            q = ( 3.*(a4_1[0, 0, -1] + d4 * a4_1) - q[0, 0, -1] ) / bet
            gam = d4 / bet
    with computation(PARALLEL):
        with interval(-1,None):
            #set bottom
            d4 = delp[0,0,-2] / delp[0,0,-1]
            a_bot = 1.+d4*(d4+1.5)
            q = (2.*d4*(d4+1.) * a4_1[0,0,-1] + a4_1[0,0,-2] - a_bot*q[0,0,-1]) / (d4*(d4+0.5) - a_bot * gam[0,0,-1])
            # q_bot = (2.*d4*(d4+1.) * a4_1 + a4_1[0,0,-1] - a_bot*q) / (d4*(d4+0.5) - a_bot * gam)
            # q = q - (gam * q_bot)
    with computation(BACKWARD), interval(0,-1):
        q = q - gam[0,0,0] * q[0,0,1]

@utils.stencil()
def set_avals(q: sd, a4_1:sd, a4_2:sd, a4_3:sd, a4_4:sd, q_bot:sd):
    with computation(PARALLEL):
        with interval(...):
            a4_2 = q
            a4_3 = q[0, 0, 1]
            a4_4 = 3.*(2.*a4_1 - (q+q[0,0,1]))
        # with interval(-1,None):
        #     a4_2 = q
        #     a4_3 = q_bot
        #     a4_4 = 3.*(2.*a4_1 - (q+q_bot))

def numpy_constraints(q, gam, a4_1, a4_2, a4_3, km, i1, i_extent, iv):
    for ii in np.array(range(i_extent))+i1:
        q[ii,0,1]=min(q[ii,0,1], max(a4_1[ii,0,1], a4_1[ii,0,0]))
        q[ii,0,1]=max(q[ii,0,1], min(a4_1[ii,0,1], a4_1[ii,0,0]))
        for kk in range(km):
            if kk>0:
                gam[ii,0,kk] = a4_1[ii,0,kk] - a4_1[ii,0,kk-1]
        for kk in range(2,km-1):
            if gam[ii,0,kk-1]*gam[ii,0,kk+1]>0:
                q[ii,0,kk]=min(q[ii,0,kk], max(a4_1[ii,0,kk], a4_1[ii,0,kk-1]))
                q[ii,0,kk]=max(q[ii,0,kk], min(a4_1[ii,0,kk], a4_1[ii,0,kk-1]))
            elif gam[ii,0,kk-1]>0:
                q[ii,0,kk]=max(q[ii,0,kk], min(a4_1[ii,0,kk], a4_1[ii,0,kk-1]))
            else:
                q[ii,0,kk]=min(q[ii,0,kk], max(a4_1[ii,0,kk], a4_1[ii,0,kk-1]))
                if iv == 0:
                    q[ii,0,kk] = max(q[ii,0,kk],0)
        q[ii,0,km-1]=min(q[ii,0,km-1], max(a4_1[ii,0,km-1], a4_1[ii,0,km-2]))
        q[ii,0,km-1]=max(q[ii,0,km-1], min(a4_1[ii,0,km-1], a4_1[ii,0,km-2]))
        for kk in range(km):
            a4_2[ii,0,kk]=q[ii,0,kk]
            a4_3[ii,0,kk]=q[ii,0,kk+1]
    return 0


@utils.stencil()
def Apply_constraints(q:sd, gam:sd, a4_1:sd, a4_2:sd, a4_3:sd, iv: int):
    with computation(PARALLEL):
        with interval(1,None):
            tmp = a4_1[0,0,-1] if a4_1[0,0,-1] > a4_1 else a4_1
            tmp2 = a4_1[0,0,-1] if a4_1[0,0,-1] < a4_1 else a4_1
        with interval(1,2):
            #do top
            # q = q if (q < a4_1[0,0,-1] or q < a4_1) else a4_1[0,0,-1] if a4_1[0,0,-1] > a4_1 else a4_1
            # q = q if (q > a4_1[0,0,-1] or q > a4_1) else a4_1[0,0,-1] if a4_1[0,0,-1] < a4_1 else a4_1
            # tmp = a4_1[0,0,-1] if a4_1[0,0,-1] > a4_1 else a4_1
            q = q if q < tmp else tmp
            # tmp2 = a4_1[0,0,-1] if a4_1[0,0,-1] < a4_1 else a4_1
            q = q if q > tmp2 else tmp2
        with interval(1, None):
            gam = a4_1 - a4_1[0,0,-1]
        with interval(2,-1):
            #do middle
            if (gam[0,0,-1]*gam[0,0,1]) >0:
                # q = q if (q < a4_1[0,0,-1] or q < a4_1) else a4_1[0,0,-1] if a4_1[0,0,-1] > a4_1 else a4_1
                # q = q if (q > a4_1[0,0,-1] or q > a4_1) else a4_1[0,0,-1] if a4_1[0,0,-1] < a4_1 else a4_1
                # tmp = a4_1[0,0,-1] if a4_1[0,0,-1] > a4_1 else a4_1
                q = q if q < tmp else tmp
                # tmp2 = a4_1[0,0,-1] if a4_1[0,0,-1] < a4_1 else a4_1
                q = q if q > tmp2 else tmp2
            elif gam[0,0,-1] > 0:
                #there's a local maximum
                # q = q if (q > a4_1[0,0,-1] or q > a4_1) else a4_1[0,0,-1] if a4_1[0,0,-1] < a4_1 else a4_1
                # tmp2 = a4_1[0,0,-1] if a4_1[0,0,-1] < a4_1 else a4_1
                q = q if q > tmp2 else tmp2
            else:
                #there's a local minimum
                # q = q if (q < a4_1[0,0,-1] or q < a4_1) else a4_1[0,0,-1] if a4_1[0,0,-1] > a4_1 else a4_1
                # tmp = a4_1[0,0,-1] if a4_1[0,0,-1] > a4_1 else a4_1
                q = q if q < tmp else tmp
                # if iv == 0:
                q = 0. if (q < 0. and iv == 0) else q
            # q = constrain_interior(q, gam, a4_1)
        with interval(-1,None):
            #do bottom
            # q = q if (q < a4_1[0,0,-1] or q < a4_1) else a4_1[0,0,-1] if a4_1[0,0,-1] > a4_1 else a4_1
            # q = q if (q > a4_1[0,0,-1] or q > a4_1) else a4_1[0,0,-1] if a4_1[0,0,-1] < a4_1 else a4_1
            #tmp = a4_1[0,0,-1] if a4_1[0,0,-1] > a4_1 else a4_1
            q = q if q < tmp else tmp
            #tmp2 = a4_1[0,0,-1] if a4_1[0,0,-1] < a4_1 else a4_1
            q = q if q > tmp2 else tmp2
        with interval(...):
            #re-set a4_2 and a4_3
            a4_2 = q
            a4_3 = q[0, 0, 1]
        # with interval(-1,None):
        #     a4_2 = q
        #     a4_3 = q_bot

@utils.stencil()
def set_extm(extm: sd, a4_1: sd, a4_2: sd, a4_3: sd, gam: sd):
    with computation(PARALLEL):
        with interval(0,1):
            extm = (a4_2 - a4_1) * (a4_3 - a4_1) > 0.
        with interval(1,-1):
            extm = gam*gam[0,0,1] < 0.
        with interval(-1,None):
            extm = (a4_2 - a4_1) * (a4_3 - a4_1) > 0.
        
@utils.stencil()
def set_exts(a4_4: sd, ext5: sd, ext6: sd, a4_1: sd, a4_2: sd, a4_3: sd):
    with computation(PARALLEL), interval(...):
        x0 = 2.0 * a4_1 - (a4_2 + a4_3)
        x1 = absolute_value(a4_2 - a4_3)
        a4_4 = 3.0 * x0
        ext5 = absolute_value(x0) > x1
        ext6 = absolute_value(a4_4) > x1


@utils.stencil()
def set_top_as_iv0(a4_1:sd, a4_2:sd, a4_3:sd, a4_4:sd):
    with computation(PARALLEL):
        with interval(0,1):
            a4_2 = a4_2 if a4_2 > 0. else 0.
        with interval(...):
            a4_4 = 3*(2*a4_1 - (a4_2 + a4_3)) 

@utils.stencil()
def set_top_as_iv1(a4_1:sd, a4_2:sd, a4_3:sd, a4_4:sd):
    with computation(PARALLEL):
        with interval(0,1):
                a4_2 = 0. if a4_2*a4_1 <= 0. else a4_2
        with interval(...):
            a4_4 = 3*(2*a4_1 - (a4_2 + a4_3))                

@utils.stencil()
def set_top_as_iv2(a4_1:sd, a4_2:sd, a4_3:sd, a4_4:sd):
    with computation(PARALLEL):
        with interval(0,1):
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.
        with interval(1,None):
            a4_4 = 3*(2*a4_1 - (a4_2 + a4_3))

@utils.stencil()
def set_top_as_else(a4_1:sd, a4_2:sd, a4_3:sd, a4_4:sd):
    with computation(PARALLEL):
        with interval(...):
            a4_4 = 3*(2*a4_1 - (a4_2 + a4_3))    

@utils.stencil()
def set_inner_as_kordsmall(a4_1:sd, a4_2:sd, a4_3:sd, a4_4:sd, gam:sd, extm:sd, ext5:sd, ext6:sd):
    with computation(PARALLEL), interval(...):
        #left edges?
        pmp_1 = a4_1 - gam[0,0,1]
        lac_1 = pmp_1 + 1.5*gam[0,0,2]
        tmp_min = a4_1 if (a4_1 < pmp_1) and (a4_1 < lac_1) else pmp_1 if pmp_1 < lac_1 else lac_1
        tmp_max0 = a4_2 if a4_2 > tmp_min else tmp_min
        tmp_max = a4_1 if (a4_1 > pmp_1) and (a4_1 > lac_1) else pmp_1 if pmp_1 > lac_1 else lac_1
        a4_2 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
        #right edges?
        pmp_2 = a4_1 + 2.* gam[0,0,1]
        lac_2 = pmp_2 - 1.5 * gam[0,0,-1]
        tmp_min = a4_1 if (a4_1 < pmp_2) and (a4_1 < lac_2) else pmp_2 if pmp_2 < lac_2 else lac_2
        tmp_max0 = a4_3 if a4_3 > tmp_min else tmp_min
        tmp_max = a4_1 if (a4_1 > pmp_2) and (a4_1 > lac_2) else pmp_2 if pmp_2 > lac_2 else lac_2
        a4_3 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
        a4_4 = 3.*(2.*a4_1 - (a4_2 + a4_3))

@gtscript.function
def crycrycry2(a4_1, a4_2, a4_3, a4_4, gam):
    pmp_1 = a4_1 - 2.* gam[0,0,1]
    lac_1 = pmp_1 + 1.5 * gam[0,0,2]
    tmp_min = a4_1 if (a4_1 < pmp_1) and (a4_1 < lac_1) else pmp_1 if pmp_1 < lac_1 else lac_1
    tmp_max0 = a4_2 if a4_2 > tmp_min else tmp_min
    tmp_max = a4_1 if (a4_1 > pmp_1) and (a4_1 > lac_1) else pmp_1 if pmp_1 > lac_1 else lac_1
    tmp_a4_2 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
    ret_val = tmp_a4_2 if absolute_value(a4_4) > absolute_value(a4_2 - a4_3) else a4_2
    return ret_val

@gtscript.function
def crycrycry3(a4_1, a4_2, a4_3, a4_4, gam):
    pmp_2 = a4_1 + 2.* gam
    lac_2 = pmp_2 - 1.5 * gam[0,0,-1]
    tmp_min = a4_1 if (a4_1 < pmp_2) and (a4_1 < lac_2) else pmp_2 if pmp_2 < lac_2 else lac_2
    tmp_max0 = a4_3 if a4_3 > tmp_min else tmp_min
    tmp_max = a4_1 if (a4_1 > pmp_2) and (a4_1 > lac_2) else pmp_2 if pmp_2 > lac_2 else lac_2
    tmp_a4_3 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
    ret_val = tmp_a4_3 if absolute_value(a4_4) > absolute_value(a4_2 - a4_3) else a4_2
    return ret_val

@utils.stencil()
def set_inner_as_kord9(a4_1:sd, a4_2:sd, a4_3:sd, a4_4:sd, gam:sd, extm:sd, ext5:sd, ext6:sd):
    with computation(PARALLEL), interval(...):
        # a4_4 = 0. if (extm == 1 and (extm[0,0,-1]==1 or extm[0,0,1]==1)) else 6.*a4_1 - 3.*(a4_2 + a4_3)
        # temp_a4_2 = a4_1 if extm == 1 and (extm[0,0,-1]==1 or extm[0,0,1]==1) else crycrycry2(a4_1, a4_2, a4_3, a4_4, gam)
        # temp_a4_3 = a4_1 if extm == 1 and (extm[0,0,-1]==1 or extm[0,0,1]==1) else crycrycry3(a4_1, a4_2, a4_3, a4_4, gam)
        # a4_2 = temp_a4_2
        # a4_3 = temp_a4_3
        # a4_4 = 6.*a4_1 - 3.*(a4_2 + a4_3) #the conditional for this is silly and long, and the else works out to this mathematically anyway, with slightly more memory accesses.
        # if not (extm == 1 and (extm[0,0,-1]==1 or extm[0,0,1]==1)) and not (absolute_value(a4_4) > absolute_value(a4_2 - a4_3))
        
        pmp_1 = a4_1 - 2.* gam[0,0,1]
        lac_1 = pmp_1 + 1.5 * gam[0,0,2]
        pmp_2 = a4_1 + 2.* gam
        lac_2 = pmp_2 - 1.5 * gam[0,0,-1]
        tmp_min = a4_1
        tmp_max = a4_2
        tmp_max0 = a4_1

        if extm and extm[0,0,-1]:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.
        elif extm and extm[0,0,1]:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.
        else:
            a4_4 = 6.*a4_1 - 3.*(a4_2 + a4_3)
            if absolute_value(a4_4) > absolute_value(a4_2 - a4_3):
                # pmp_1 = a4_1 - 2.* gam[0,0,1]
                # lac_1 = pmp_1 + 1.5 * gam[0,0,2]
                tmp_min = a4_1 if (a4_1 < pmp_1) and (a4_1 < lac_1) else pmp_1 if pmp_1 < lac_1 else lac_1
                tmp_max0 = a4_2 if a4_2 > tmp_min else tmp_min
                tmp_max = a4_1 if (a4_1 > pmp_1) and (a4_1 > lac_1) else pmp_1 if pmp_1 > lac_1 else lac_1
                a4_2 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
                # pmp_2 = a4_1 + 2.* gam
                # lac_2 = pmp_2 - 1.5 * gam[0,0,-1]
                tmp_min = a4_1 if (a4_1 < pmp_2) and (a4_1 < lac_2) else pmp_2 if pmp_2 < lac_2 else lac_2
                tmp_max0 = a4_3 if a4_3 > tmp_min else tmp_min
                tmp_max = a4_1 if (a4_1 > pmp_2) and (a4_1 > lac_2) else pmp_2 if pmp_2 > lac_2 else lac_2
                a4_3 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
                a4_4 = 6.*a4_1 - 3.*(a4_2 + a4_3)
            else:
                a4_2 = a4_2

@utils.stencil()
def set_inner_as_kord10(a4_1:sd, a4_2:sd, a4_3:sd, a4_4:sd, gam:sd, extm:sd, ext5:sd, ext6:sd):
    with computation(PARALLEL), interval(...):
        # a4_2 = a4_1 if (ext5==1 and (ext5[0,0,-1]==1 or ext5[0,0,1]==1)) else crycrycry2(a4_1, a4_2, a4_3, a4_4, gam) if ((ext5==1 and (ext6[0,0,-1]==1 or ext6[0,0,1]==1)) or (ext6==1 and (ext5[0,0,-1]==1 or ext5[0,0,1]==1))) else a4_2
        # a4_3 = a4_1 if (ext5 and (ext5[0,0,-1] or ext5[0,0,1])) else crycrycry3(a4_1, a4_2, a4_3, a4_4, gam) if ((ext5 and (ext6[0,0,-1] or ext6[0,0,1])) or (ext6 and (ext5[0,0,-1] or ext5[0,0,1]))) else a4_3
        pmp_1 = a4_1 - 2.*gam[0,0,1]
        lac_1 = pmp_1 + 1.5 * gam[0,0,2]
        pmp_2 = a4_1 + 2. * gam
        lac_2 = pmp_2 - 1.5 * gam[0,0,-1]
        tmp_min2 = a4_1 if (a4_1 < pmp_1) and (a4_1 < lac_1) else pmp_1 if pmp_1 < lac_1 else lac_1
        tmp_max2 = a4_1 if (a4_1 > pmp_1) and (a4_1 > lac_1) else pmp_1 if pmp_1 > lac_1 else lac_1
        tmp2 = a4_2 if a4_2 > tmp_min2 else tmp_min2

        tmp_min3 = a4_1 if (a4_1 < pmp_2) and (a4_1 < lac_2) else pmp_2 if pmp_2 < lac_2 else lac_2
        tmp_max3 = a4_1 if (a4_1 > pmp_2) and (a4_1 > lac_2) else pmp_2 if pmp_2 > lac_2 else lac_2
        tmp3 = a4_3 if a4_3 > tmp_min3 else tmp_min3
        if ext5:
            if ext5[0,0,-1] or ext5[0,0,1]:
                a4_2 = a4_1
                a4_3 = a4_1
            elif ext6[0,0,-1] or ext6[0,0,1]:
                # pmp_1 = a4_1 -2.*gam[0,0,1]
                # lac_1 = pmp_1 + 1.5 * gam[0,0,2]
                # tmp_min = a4_1 if (a4_1 < pmp_1) and (a4_1 < lac_1) else pmp_1 if pmp_1 < lac_1 else lac_1
                # tmp_max = a4_1 if (a4_1 > pmp_1) and (a4_1 > lac_1) else pmp_1 if pmp_1 > lac_1 else lac_1
                # tmp_max0 = a4_2 if a4_2 > tmp_min else tmp_min
                a4_2 = tmp2 if tmp2 < tmp_max2 else tmp_max2
                # pmp_2 = a4_1 + 2. * gam
                # lac_2 = pmp_2 - 1.5 * gam[0,0,-1]
                # tmp_min = a4_1 if (a4_1 < pmp_2) and (a4_1 < lac_2) else pmp_2 if pmp_2 < lac_2 else lac_2
                # tmp_max0 = a4_3 if a4_3 > tmp_min else tmp_min
                # tmp_max = a4_1 if (a4_1 > pmp_2) and (a4_1 > lac_2) else pmp_2 if pmp_2 > lac_2 else lac_2
                a4_3 = tmp3 if tmp3 < tmp_max3 else tmp_max3
            else:
                a4_2 = a4_2
        elif ext6:
            if ext5[0,0,-1] or ext5[0,0,1]:
                # pmp_1 = a4_1 - 2. * gam[0,0,1]
                # lac_1 = pmp_1 + 1.5*gam[0,0,2]
                # tmp_min = a4_1 if (a4_1 < pmp_1) and (a4_1 < lac_1) else pmp_1 if pmp_1 < lac_1 else lac_1
                # tmp_max0 = a4_2 if a4_2 > tmp_min else tmp_min
                # tmp_max = a4_1 if (a4_1 > pmp_1) and (a4_1 > lac_1) else pmp_1 if pmp_1 > lac_1 else lac_1
                # a4_2 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
                a4_2 = tmp2 if tmp2 < tmp_max2 else tmp_max2
                # pmp_2 = a4_1 + 2. * gam
                # lac_2 = pmp_2 - 1.5 * gam[0,0,-1]
                # tmp_min = a4_1 if (a4_1 < pmp_2) and (a4_1 < lac_2) else pmp_2 if pmp_2 < lac_2 else lac_2
                # tmp_max0 = a4_3 if a4_3 > tmp_min else tmp_min
                # tmp_max = a4_1 if (a4_1 > pmp_2) and (a4_1 > lac_2) else pmp_2 if pmp_2 > lac_2 else lac_2
                # a4_3 = tmp_max0 if tmp_max0 < tmp_max else tmp_max
                a4_3 = tmp3 if tmp3 < tmp_max3 else tmp_max3
            else:
                a4_2 = a4_2
        else:
            a4_2 = a4_2
        a4_4 = 6.*a4_1 - 3.*(a4_2 + a4_3)

@utils.stencil()
def set_bottom_as_iv0(a4_1:sd, a4_2:sd, a4_3:sd, a4_4:sd):
    with computation(PARALLEL):
        with interval(1,None):
            a4_3 = a4_3 if a4_3 > 0. else 0.
        with interval(...):
            a4_4 = 3.*(2.*a4_1 - (a4_2+a4_3))

@utils.stencil()
def set_bottom_as_iv1(a4_1:sd, a4_2:sd, a4_3:sd, a4_4:sd):
    with computation(PARALLEL):
        with interval(1,None):
            a4_3 = 0. if a4_3 * a4_1 <= 0. else a4_3
        with interval(...):
            a4_4 = 3.*(2.*a4_1 - (a4_2+a4_3))

@utils.stencil()
def set_bottom_as_else(a4_1:sd, a4_2:sd, a4_3:sd, a4_4:sd):
    with computation(PARALLEL):
        with interval(...):
            a4_4 = 3.0 * (2.0 * a4_1 - (a4_2 + a4_3))


def compute(qs, a4_1, a4_2, a4_3, a4_4, delp, set_gam, set_q, set_a4, b_q, b_gam, b_a4, b_extm, b_ext5, b_ext6,cs1_extm,cs1_a4_1,cs1_a4_2,cs1_a4_3,cs1_a4_4,cs1b_a4_1,cs1b_a4_2,cs1b_a4_3,cs1b_a4_4,cs2_extm,cs2_a4_1,cs2_a4_2,cs2_a4_3,cs2_a4_4,cs2b_a4_1,cs2b_a4_2,cs2b_a4_3,cs2b_a4_4,huy_a4_1,huy_a4_2,huy_a4_3,huy_a4_4, km, i1, i2, iv, kord):
    # TODO: how do we handle 2d-stencils/take a 2d slice of a 3d array?
    # TODO: how do we handle loopy stencils, e.g. q(i,k) = q(i,k) - gam(i,k+1)*q(i,k+1)?
    # Or q(i,k) = (3.*(a4(1,i,k-1)+a4(1,i,k)) - q(i,k-1))/bet?
    # qs is 1-d
    # delp is 2d
    # a4 is 3d but weirdly shaped, probably only care about i=1?
    # TODO how put these all together??
    i_extent = i2 - i1 + 1
    
    grid = spec.grid
    orig = (i1, 0, 0)
    full_orig = (grid.is_, 0, 0)
    dom = (i_extent,1,km)
    print(iv, kord)
    # ext_dom = (i_extent,1,km+1)
    # extend_shape =  list(delp.shape)
    # extend_shape[-1]+=1
    gam = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    q = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    q_bot = utils.make_storage_from_shape(delp.shape, origin=full_orig)

    extm = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    ext5 = utils.make_storage_from_shape(delp.shape, origin=full_orig)
    ext6 = utils.make_storage_from_shape(delp.shape, origin=full_orig)

    if iv==-2:
        set_vals_2(gam, q, delp, a4_1, q_bot, qs, origin=orig, domain=dom)
    else:
        set_vals_1(gam, q, delp, a4_1, q_bot, origin=orig, domain=(i_extent, 1, km+1))

    # for ii in range(km):
    #     if (set_q[i1:i2,:,ii]==q[i1:i2,:,ii]).all():
    #         print(ii)
    #     else:
    #         #print("yikes")
    #         print(max(set_q[i1:i2,0,ii]-q[i1:i2,0,ii]))
    # assert (set_q[i1:i2,:,:] == q[i1:i2,:,:]).all()
    if abs(kord) > 16:
        set_avals(q, a4_1, a4_2, a4_3, a4_4, q_bot, origin=orig, domain=dom)
    else:
        Apply_constraints(q, gam, a4_1, a4_2, a4_3, iv, origin=orig, domain=dom)
        # p0 = numpy_constraints(q, gam, a4_1, a4_2, a4_3, km, i1, i_extent, iv)
        set_extm(extm, a4_1, a4_2, a4_3, gam, origin=orig, domain=dom)

        if abs(kord) > 9:
            set_exts(a4_4, ext5, ext6, a4_1, a4_2, a4_3, origin=orig, domain=dom)


        # for ii in range(km):
        #     if (b_gam[i1:i2,0,ii]==gam[i1:i2,0,ii]).all():
        #         print(ii)
        #     else:
        #         #print("yikes")
        #         print(max(gam[i1:i2,0,ii]-gam[i1:i2,0,ii]))

        # for ii in range(km):
        #     if (b_q[i1:i2,:,ii]==q[i1:i2,:,ii]).all():
        #         print(ii)
        #     else:
        #         #print("yikes")
        #         print(max(b_q[i1:i2,0,ii]-q[i1:i2,0,ii]))

        # print(b_a4.shape)

        # examine = (b_q[i1:i2,0,31] != q[i1:i2,0,31])

        # print(gam[i1:i2,0,61][examine])
        # print(gam[i1:i2,0,59][examine])
        # print(a4_1[i1:i2,0,60][examine])
        # print(a4_1[i1:i2,0,59][examine])
        # print(q[i1:i2,0,60][examine])
        # print(b_q[i1:i2,0,60][examine])

        # assert (b_a4[0,i1:i2,:]==a4_1[i1:i2,0,:]).all()
        # assert (b_gam[i1:i2,:,:] == gam[i1:i2,:,:]).all()
        # assert (b_q[i1:i2,:,:] == q[i1:i2,:,:]).all()
        # assert (b_a4[1,i1:i2,:]==a4_2[i1:i2,0,:]).all()
        # assert (b_a4[2,i1:i2,:]==a4_3[i1:i2,0,:]).all()
        # assert (b_a4[3,i1:i2,:]==a4_4[i1:i2,0,:]).all()
        # assert (b_extm[i1:i2,:,:] == extm[i1:i2,:,:]).all()
        # assert (b_ext5[i1:i2,:,:] == ext5[i1:i2,:,:]).all()
        # assert (b_ext6[i1:i2,:,:] == ext6[i1:i2,:,:]).all()

        if iv == 0:
            set_top_as_iv0(a4_1, a4_2, a4_3, a4_4, origin=orig, domain=(i_extent, 1, 2))
            assert (cs1_a4_1[i1:i2,0,:]==a4_1[i1:i2,0,:]).all()
            assert (cs1_a4_2[i1:i2,0,:]==a4_2[i1:i2,0,:]).all()
            assert (cs1_a4_3[i1:i2,0,:]==a4_3[i1:i2,0,:]).all()
            assert (cs1_a4_4[i1:i2,0,:]==a4_4[i1:i2,0,:]).all()
            a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, 0, 1)
            print(a4_2[i1:i2,0,0][cs1b_a4_2[i1:i2,0,0]!=a4_2[i1:i2,0,0]])
            print(cs1b_a4_2[i1:i2,0,0][cs1b_a4_2[i1:i2,0,0]!=a4_2[i1:i2,0,0]].shape)
            print(cs1b_a4_2[i1:i2,0,0][cs1b_a4_2[i1:i2,0,0]!=a4_2[i1:i2,0,0]])
            print(a4_1[i1:i2,0,0][cs1b_a4_2[i1:i2,0,0]!=a4_2[i1:i2,0,0]])
            assert (cs1b_a4_1[i1:i2,0,0]==a4_1[i1:i2,0,0]).all()
            assert (cs1b_a4_2[i1:i2,0,0]==a4_2[i1:i2,0,0]).all()
            assert (cs1b_a4_3[i1:i2,0,0]==a4_3[i1:i2,0,0]).all()
            assert (cs1b_a4_4[i1:i2,0,0]==a4_4[i1:i2,0,0]).all()
        elif iv == -1:
            set_top_as_iv1(a4_1, a4_2, a4_3, a4_4, origin=orig, domain=(i_extent, 1, 2))
            assert (cs1_a4_1[i1:i2,0,:]==a4_1[i1:i2,0,:]).all()
            assert (cs1_a4_2[i1:i2,0,:]==a4_2[i1:i2,0,:]).all()
            assert (cs1_a4_3[i1:i2,0,:]==a4_3[i1:i2,0,:]).all()
            assert (cs1_a4_4[i1:i2,0,:]==a4_4[i1:i2,0,:]).all()
            a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, 0, 1)
            print(a4_2[i1:i2,0,0][cs1b_a4_2[i1:i2,0,0]!=a4_2[i1:i2,0,0]])
            print(cs1b_a4_2[i1:i2,0,0][cs1b_a4_2[i1:i2,0,0]!=a4_2[i1:i2,0,0]].shape)
            print(cs1b_a4_2[i1:i2,0,0][cs1b_a4_2[i1:i2,0,0]!=a4_2[i1:i2,0,0]])
            print(a4_1[i1:i2,0,0][cs1b_a4_2[i1:i2,0,0]!=a4_2[i1:i2,0,0]])
            assert (cs1b_a4_1[i1:i2,0,:]==a4_1[i1:i2,0,:]).all()
            assert (cs1b_a4_2[i1:i2,0,:]==a4_2[i1:i2,0,:]).all()
            assert (cs1b_a4_3[i1:i2,0,:]==a4_3[i1:i2,0,:]).all()
            assert (cs1b_a4_4[i1:i2,0,:]==a4_4[i1:i2,0,:]).all()
        elif iv == 2:
            set_top_as_iv2(a4_1, a4_2, a4_3, a4_4, origin=orig, domain=(i_extent, 1, 2))
        else:
            set_top_as_else(a4_1, a4_2, a4_3, a4_4, origin=orig, domain=(i_extent, 1, 2))
            a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, 0, 1)
        a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(a4_1, a4_2, a4_3, a4_4, extm, 2, i1, i_extent, 1, 1)

        if abs(kord) < 9:
            set_inner_as_kordsmall(a4_1, a4_2, a4_3, a4_4, gam, extm, ext5, ext6, origin=(i1,0,2), domain=(i_extent, 1, km-4))
        elif abs(kord) == 9:
            set_inner_as_kord9(a4_1, a4_2, a4_3, a4_4, gam, extm, ext5, ext6, origin=(i1,0,2), domain=(i_extent, 1, km-4))
        elif abs(kord) == 10:
            set_inner_as_kord10(a4_1, a4_2, a4_3, a4_4, gam, extm, ext5, ext6, origin=(i1,0,2), domain=(i_extent, 1, km-4))
        else:
            print("kord {0} not implemented yet. Go bug a dev for it.".format(kord))

        if iv == 0:
            a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, 2, km-4)
        
        if iv == 0:
            set_bottom_as_iv0(a4_1, a4_2, a4_3, a4_4, origin=(i1,0,km-1), domain=(i_extent, 1, 2))
        elif iv == -1:
            set_bottom_as_iv1(a4_1, a4_2, a4_3, a4_4, origin=(i1,0,km-1), domain=(i_extent, 1, 2))
        else:
            set_bottom_as_else(a4_1, a4_2, a4_3, a4_4, origin=(i1,0,km-1), domain=(i_extent, 1, 2))
        a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(a4_1, a4_2, a4_3, a4_4, extm, 2, i1, i_extent, km-2, 1)
        a4_1, a4_2, a4_3, a4_4 = cs_limiters.compute(a4_1, a4_2, a4_3, a4_4, extm, 1, i1, i_extent, km-1, 1)

    return a4_1, a4_2, a4_3, a4_4
