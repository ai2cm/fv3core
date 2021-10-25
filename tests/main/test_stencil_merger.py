from os import name
from typing import Any, Callable, Optional, Sequence, List, Tuple, Union

import numpy as np
import pytest

from eve import Node, NodeVisitor
from gt4py.gtscript import PARALLEL, computation, interval, stencil
from gtc import common
from gt4py.storage import Storage

import fv3core._config as spec
from fv3core.decorators import StencilInterface, disable_merge_stencils, enable_merge_stencils
from fv3core.stencils.riem_solver_c import RiemannSolverC
from fv3core.utils.global_config import set_backend
from fv3core.utils.gt4py_utils import make_storage_from_shape_uncached
from fv3core.utils.mpi import MPI
from fv3core.utils.typing import FloatField


def copy_stencil(q_in: FloatField, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_in


def add1_stencil(q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out += 1.0


def setup_data_vars(num_storages: int = 3, init_val: float = 1.0):
    shape = (7, 7, 3)
    storages = []
    for n in range(num_storages):
        storage = make_storage_from_shape_uncached(shape)
        storage[:] = init_val
        storages.append(storage)

    return tuple(storages)


@pytest.mark.sequential
@pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.Get_size() > 1,
    reason="Running in parallel with mpi",
)
@pytest.mark.parametrize("backend", ("numpy", "gtx86"))
@pytest.mark.parametrize("rebuild", (True, False))
@pytest.mark.parametrize("do_merge", (False, True))
def test_stencil_merger(backend: str, rebuild: bool, do_merge: bool):
    set_backend(backend)

    q_in, q_out, q_ref = setup_data_vars()
    q_ref[1:3, 1:3, :] = 2.0

    origin = (1, 1, 0)
    domain = (2, 2, 3)

    q_out = run_stencil_test(q_in, q_out, origin, domain, backend, rebuild, do_merge)
    assert np.array_equal(q_out, q_ref)


def run_stencil_test(
    q_in: FloatField,
    q_out: FloatField,
    origin: Tuple[int, ...],
    domain: Tuple[int, ...],
    backend: str,
    rebuild: bool,
    do_merge: bool,
):
    if do_merge:
        enable_merge_stencils()

    copy_object = stencil(
        definition=copy_stencil,
        backend=backend,
        rebuild=rebuild,
    )

    add1_object = stencil(
        definition=add1_stencil,
        backend=backend,
        rebuild=rebuild,
    )

    if do_merge:
        disable_merge_stencils()

    copy_object(q_in, q_out, origin=origin, domain=domain)
    add1_object(q_out, origin=origin, domain=domain)

    return q_out


class GraphNode(common.LocNode):
    name: str

class ComputeNode(GraphNode):
    computation: Callable


class StorageNode(GraphNode):
    storage: Any


class ScalarNode(GraphNode):
    value: Union[int, float, str]


class FlowGraph(object):  # common.LocNode, common.SymbolTableTrait):
    def __init__(
        self,
        name: str = "",
        compute_nodes: List[ComputeNode] = (),
        storage_nodes: List[StorageNode] = (),
        scalar_nodes: List[ScalarNode] = (),
    ):
        self.name = name
        self.compute_nodes = list(compute_nodes)
        self.storage_nodes = list(storage_nodes)
        self.scalar_nodes = list(scalar_nodes)

    def __setitem__(self, name: str, member: Any) -> None:
        if isinstance(member, Storage):
            # New storage node
            storage_node = StorageNode(name=name, storage=member)
            self.storage_nodes.append(storage_node)
        elif isinstance(member, StencilInterface):
            # New compute node
            compute_node = ComputeNode(name=name, computation=member)
            self.compute_nodes.append(compute_node)
        elif isinstance(member, (int, float, str)):
            # New scalar node
            scalar_node = ScalarNode(name=name, value=member)
            self.scalar_nodes.append(scalar_node)


class StencilClassVisitor(NodeVisitor):
    def visit(self, stencil_class: Callable) -> None:
        self._flow_graph = FlowGraph(name=stencil_class.__class__.__name__)
        for name, member in stencil_class.__dict__.items():
            self._flow_graph[name] = member
        return self._flow_graph

    @classmethod
    def apply(cls, stencil_class: Callable) -> FlowGraph:
        instance = cls()
        return instance.visit(stencil_class)


@pytest.mark.parametrize("backend", ("numpy",))  # , "gtx86"))
@pytest.mark.parametrize("rebuild", (False,),)  #  True))
def test_stencil_class(backend: str, rebuild: bool):
    set_backend(backend)
    compute_func = RiemannSolverC(spec.grid, spec.namelist.p_fac)
    assert isinstance(compute_func, Callable)

    flow_graph = StencilClassVisitor.apply(compute_func)
    assert flow_graph.name == "RiemannSolverC"

    assert len(flow_graph.scalar_nodes) == 1
    assert flow_graph.scalar_nodes[0].name == "_pfac"
    assert flow_graph.scalar_nodes[0].value == 0.0

    assert len(flow_graph.compute_nodes) == 3
    assert len(flow_graph.storage_nodes) == 7
