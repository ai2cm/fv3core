from gt4py.gtscript import FORWARD, computation, interval

from fv3core.utils.typing import FloatField


def edge_pe(pe: FloatField, delp: FloatField, ptop: float):
    """
    This corresponds to the pe_halo routine in FV3core
    Updading the interface pressure from the pressure differences
    Arguments:
        pe: The pressure on the interfaces of the cell
        delp: The pressure difference between vertical grid cells
        ptop: The pressure level at the top of the grid
    """

    with computation(FORWARD):
        with interval(0, 1):
            pe[0, 0, 0] = ptop
        with interval(1, None):
            pe[0, 0, 0] = pe[0, 0, -1] + delp[0, 0, -1]
