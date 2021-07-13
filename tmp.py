import dace
import numpy as np
from dace.frontend.python.common import SDFGConvertible
from gt4py.gtscript import SDFGWrapper, Field, stencil
from types import SimpleNamespace

origin=(2,0,0)
domain= (2,2,2)

class TillCallable(SDFGConvertible):

    def __init__(self, definition):
        self.definition = definition
        self.stencil_object = stencil(backend="gtc:dace")(definition)

    def __call__(self, grid, *args, **kwargs):
        self.stencil_object(*args, **kwargs, origin=grid.origin, domain=grid.domain)

    def __sdfg__(self, grid, *args, **kwargs):
        self.sdfg_wrapper = SDFGWrapper(definition=self.definition, origin=grid.origin, domain=grid.domain)
        return self.sdfg_wrapper.__sdfg__(self, *args, **kwargs)

    def __sdfg_signature__(self):
        return ( ['grid']+[arg for arg in self.definition.__annotations__.keys() if arg != 'return'], ['grid'])


    def __sdfg_closure__(self, *args, **kwargs):
        return {}

def tilldecorator(func):

    return TillCallable(definition=func)

@tilldecorator
def a_stencil(field: Field[np.float64]):
    with computation(PARALLEL), interval(...):
        field = 7.0

A = np.ones((10,10,10))
grid = SimpleNamespace(origin=origin, domain=domain)


@dace.program # the real test if this works both commented and not.
def forward_call(A):
    a_stencil(grid, A)

forward_call(A)

assert np.allclose(7.0, A[origin[0]:origin[0]+domain[0], origin[1]:origin[1]+domain[1], origin[2]:origin[2]+domain[2]])