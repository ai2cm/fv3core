from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import pytest

import gt4py
from eve import Node, NodeVisitor
from gt4py.definitions import AccessKind
from gt4py.gtscript import PARALLEL, computation, interval, stencil
from gtc import common
from gt4py.storage import Storage, from_array

import fv3core._config as spec
from fv3core.decorators import StencilInterface, disable_merge_stencils, enable_merge_stencils, set_flow_graph
from fv3core.stencils.riem_solver_c import RiemannSolverC
from fv3core.utils.global_config import set_backend, set_rebuild
from fv3core.utils.gt4py_utils import deserialize, make_storage_from_shape_uncached
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
        compute_nodes: Dict[str, ComputeNode] = {},
        storage_nodes: Dict[str, StorageNode] = {},
        scalar_nodes: Dict[str, ScalarNode] = {},
    ):
        self.name = name
        self.compute_nodes = dict(compute_nodes)
        self.storage_nodes = dict(storage_nodes)
        self.scalar_nodes = dict(scalar_nodes)
        self.read_edges: Dict[str, str] = {}
        self.write_edges: Dict[str, str] = {}

    def __getitem__(self, index: Any):
        if isinstance(index, str):
            name = index
            if name in self.compute_nodes:
                return self.compute_nodes[name]
            if name in self.storage_nodes:
                return self.storage_nodes[name]
            if name in self.scalar_nodes:
                return self.scalar_nodes[name]
            raise KeyError(f"No known node with name '{name}'")
        elif isinstance(index, gt4py.StencilObject):
            for compute_node in self.compute_nodes.values():
                if compute_node.computation.stencil_object is index:
                    return compute_node
        elif isinstance(index, Storage):
            for storage_node in self.storage_nodes.values():
                if storage_node.storage is index:
                    return storage_node
        else:
            for scalar_node in self.scalar_nodes.values():
                if scalar_node.value == index:
                    return scalar_node
        # raise KeyError(f"No known node with index id {id(index)}")
        return None

    def __setitem__(self, name: str, member: Any) -> None:
        if isinstance(member, Storage):
            # New storage node
            self.storage_nodes[name] = StorageNode(name=name, storage=member)
        elif isinstance(member, StencilInterface):
            # New compute node
            self.compute_nodes[name] = ComputeNode(name=name, computation=member)
        elif isinstance(member, (int, float, str)):
            # New scalar node
            self.scalar_nodes[name] = ScalarNode(name=name, value=member)

    def add_edge(self, compute_node: ComputeNode, data_node: Union[StorageNode, ScalarNode], is_write: bool) -> None:
        if is_write:
            edges = self.write_edges
            from_node = compute_node.name
            to_node = data_node.name
        else:
            edges = self.read_edges
            from_node = data_node.name
            to_node = compute_node.name
        if from_node not in edges:
            edges[from_node] = []
        edges[from_node].append(to_node)

    def update(self, stencil_object: gt4py.StencilObject, **kwargs: Any):
        compute_node = self[stencil_object]
        for key, item in kwargs.items():
            is_write: bool = False
            if key in stencil_object.field_info:
                data_node = self[item]
                field_info = stencil_object.field_info[key]
                is_write = bool(field_info.access & AccessKind.WRITE)
            else:
                data_node = self[item]
            if data_node is not None:
                self.add_edge(compute_node, data_node, is_write)

    def _count_edges(self, edges: Dict[str, str]) -> int:
        num_edges: int = 0
        for from_edge in edges:
            num_edges += len(edges[from_edge])
        return num_edges

    @property
    def num_read_edges(self) -> int:
        return self._count_edges(self.read_edges)

    @property
    def num_write_edges(self) -> int:
        return self._count_edges(self.write_edges)

class StencilClassVisitor(NodeVisitor):
    def visit(self, stencil_class: Callable, **kwargs: Any) -> None:
        self._flow_graph = FlowGraph(name=stencil_class.__class__.__name__)
        for name, member in kwargs.items():
            self._flow_graph[name] = member
        for name, member in stencil_class.__dict__.items():
            self._flow_graph[name] = member
        return self._flow_graph

    @classmethod
    def apply(cls, stencil_class: Callable, **kwargs: Any) -> FlowGraph:
        instance = cls()
        return instance.visit(stencil_class, **kwargs)


def mask_from_shape(shape: tuple) -> tuple:
    if len(shape) == 1:
        return (False, False, True)
    return (True,) * len(shape) + (False,) * (3 - len(shape))


def arrays_to_storages(arrays: Dict[str, Any], backend: str) -> dict:
    return {
        name: from_array(
            data=item,
            backend=backend,
            default_origin=[0] * len(item.shape),
            shape=item.shape,
            mask=mask_from_shape(item.shape),
            managed_memory=True,
        )
        if item.shape
        else item[()]
        for name, item in arrays.items()
    }


@pytest.mark.parametrize("backend", ("numpy",))  # , "gtx86"))
@pytest.mark.parametrize("rebuild", (False,),)  #  True))
def test_stencil_class(backend: str, rebuild: bool):
    set_backend(backend)
    set_rebuild(rebuild)

    compute_func = RiemannSolverC(spec.grid, spec.namelist.p_fac)
    assert isinstance(compute_func, Callable)

    call_data = deserialize("./riem_solver_c")
    call_kwargs = arrays_to_storages(call_data, backend)

    flow_graph = StencilClassVisitor.apply(compute_func, **call_kwargs)
    set_flow_graph(flow_graph)
    compute_func(**call_kwargs)

    assert flow_graph.name == "RiemannSolverC"
    assert len(flow_graph.scalar_nodes) == 3
    assert len(flow_graph.compute_nodes) == 3
    assert len(flow_graph.storage_nodes) == 16

    assert flow_graph.num_read_edges == 19
    assert flow_graph.num_write_edges == 11
