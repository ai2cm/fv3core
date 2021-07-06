# # import gt4py.gtscript
# # from fv3core.utils.typing import FloatField, FloatFieldIJ
# #
# # @gt4py.gtscript.stencil(backend="gtc:dace", rebuild=True)
# # def update_vorticity(
# #     uc: FloatField,
# #     vc: FloatField,
# #     dxc: FloatFieldIJ,
# #     dyc: FloatFieldIJ,
# #     vort_c: FloatField,
# #     fy: FloatField,
# # ):
# #     """Update vort_c.
# #
# #     Args:
# #         uc: x-velocity on C-grid (input)
# #         vc: y-velocity on C-grid (input)
# #         dxc: grid spacing in x-dir (input)
# #         dyc: grid spacing in y-dir (input)
# #         vort_c: C-grid vorticity (output)
# #     """
# #
# #     with computation(PARALLEL), interval(...):
# #         fx = dxc * uc
# #         fy = dyc * vc
# #     with computation(PARALLEL), interval(...):
# #         vort_c = fx[0, -1, 0] - fx - fy[-1, 0, 0] + fy
# #
# import dace
#
# @dace.program
# def ahoj(field):
#     field[...] = 1.0
#
# class BClass:
#
#     @dace.method
#     def __call__(self, field):
#         ahoj(field)
#
# class AClass:
#     def __init__(self):
#         self.bclass = BClass()
#         self.bclass2 = BClass()
#         self.bclass3 = BClass()
#     def __call__(self, field):
#         self.bclass(field)
#         self.bclass2(field)
#         self.bclass3(field)
#
# aclass = AClass()
# import numpy as np
# A = np.ones((3,))
# aclass(A)
#

import numpy as np
import dace
from fv3core.utils.gt4py_utils import computepath_method
class Ahoj:

    def __init__(self):
        self.asdfg = np.ones((10,))

    @computepath_method(use_dace=True)
    def methode(self, A):
        A[...] = self.asdfg + 2.0

    @computepath_method(use_dace=True)
    def __call__(self, A):
        self.methode(A)

B = np.ones((10,))
ahoj = Ahoj()
ahoj(B)
np.testing.assert_allclose(3.0, B)