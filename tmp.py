# import gt4py.gtscript
# from fv3core.utils.typing import FloatField, FloatFieldIJ
#
# @gt4py.gtscript.stencil(backend="gtc:dace", rebuild=True)
# def update_vorticity(
#     uc: FloatField,
#     vc: FloatField,
#     dxc: FloatFieldIJ,
#     dyc: FloatFieldIJ,
#     vort_c: FloatField,
#     fy: FloatField,
# ):
#     """Update vort_c.
#
#     Args:
#         uc: x-velocity on C-grid (input)
#         vc: y-velocity on C-grid (input)
#         dxc: grid spacing in x-dir (input)
#         dyc: grid spacing in y-dir (input)
#         vort_c: C-grid vorticity (output)
#     """
#
#     with computation(PARALLEL), interval(...):
#         fx = dxc * uc
#         fy = dyc * vc
#     with computation(PARALLEL), interval(...):
#         vort_c = fx[0, -1, 0] - fx - fy[-1, 0, 0] + fy
#
import dace

@dace.program
def ahoj(field):
    field[...] = 1.0

class BClass:

    @dace.method
    def __call__(self, field):
        ahoj(field)

class AClass:
    def __init__(self):
        self.bclass = BClass()
        self.bclass2 = BClass()
        self.bclass3 = BClass()
    def __call__(self, field):
        self.bclass(field)
        self.bclass2(field)
        self.bclass3(field)

aclass = AClass()
import numpy as np
A = np.ones((3,))
aclass(A)

