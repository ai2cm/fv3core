import dace
from typing import List, Tuple
from dace.transformation.helpers import get_parent_map


def nanczek(sdfg: dace.SDFG):
    """
    Insert after sdfg.simplify(...)
    """
    import copy
    from dace.sdfg import graph as gr, utils as sdutil
    from dace import data as dt, symbolic, subsets as sbs
    from dace.dtypes import ScheduleType
    import sympy as sp

    # Adds a NaN checker after every mapexit->access node
    checks: List[
        Tuple[dace.SDFGState, dace.nodes.AccessNode, gr.MultiConnectorEdge[dace.Memlet]]
    ] = []
    allmaps = [
        (me, state)
        for me, state in sdfg.all_nodes_recursive()
        if isinstance(me, dace.nodes.MapEntry)
    ]
    topmaps = [
        (me, state) for me, state in allmaps if get_parent_map(state, me) is None
    ]

    varname_whitelist = ["pt"]

    def in_varname_whitelist(var: str):
        for allowed_var in varname_whitelist:
            if var in allowed_var:
                return True
        return False

    for me, state in topmaps:
        mx = state.exit_node(me)
        for e in state.out_edges(mx):
            if isinstance(e.dst, dace.nodes.AccessNode):
                if isinstance(e.dst.desc(state.parent), dt.View):  # Skip views for now
                    continue
                node = sdutil.get_last_view_node(state, e.dst)
                if "pt" not in node.data:
                    continue
                if state.memlet_path(e)[
                    0
                ].data.dynamic:  # Skip dynamic (region) outputs
                    continue

                checks.append((state, node, e))
    for state, node, e in checks:
        # Append node that will go after the map
        newnode: dace.nodes.AccessNode = copy.deepcopy(node)
        # Move all outgoing edges to new node
        for oe in list(state.out_edges(node)):
            state.remove_edge(oe)
            state.add_edge(newnode, oe.src_conn, oe.dst, oe.dst_conn, oe.data)

        # Add map in between node and newnode
        sdfg = state.parent
        inparr = sdfg.arrays[newnode.data]
        index_expr = ", ".join(["__i%d" % i for i in range(len(inparr.shape))])
        index_printf = ", ".join(["%d"] * len(inparr.shape))
        symbols_printf = ", ".join(
            [f"{s}=%d" for s, v in sorted(sdfg.symbols.items()) if v is dace.int32]
            + [f"{s}=%lld" for s, v in sorted(sdfg.symbols.items()) if v is dace.int64]
            + [f"{s}=%u" for s, v in sorted(sdfg.symbols.items()) if v is dace.uint32]
            + [f"{s}=%llu" for s, v in sorted(sdfg.symbols.items()) if v is dace.uint64]
        )
        symbols_expr = ", ".join(
            [f"{s}" for s, v in sorted(sdfg.symbols.items()) if v is dace.int32]
            + [f"{s}" for s, v in sorted(sdfg.symbols.items()) if v is dace.int64]
            + [f"{s}" for s, v in sorted(sdfg.symbols.items()) if v is dace.uint32]
            + [f"{s}" for s, v in sorted(sdfg.symbols.items()) if v is dace.uint64]
        )
        if not symbols_expr:
            symbols_printf = "%s"
            symbols_expr = '"N/A"'
        # Get range from memlet (which may not be the entire array size)
        def evaluate(expr):
            return expr.subs({sp.Function("int_floor"): symbolic.int_floor})

        ranges = []
        for i, (b, e, s) in enumerate(e.data.subset):
            ranges.append((f"__i{i}", (evaluate(b), evaluate(e), evaluate(s))))
        k_below_79 = ""
        if len(ranges) == 3:
            k_below_79 = "&& __i2 < 50 && __i2 > 20"

        bounds_conditional = (
            f"&& __i0 >= 3 && __i0 <= 15 && __i1 >= 3 && __i1 <=15 {k_below_79}"
        )

        state.add_mapped_tasklet(
            name="nancheck",
            map_ranges=ranges,
            inputs={"__inp": dace.Memlet.simple(newnode.data, index_expr)},
            code=f"""
            if (__inp != __inp ) {{
                printf("NaN value found at {newnode.data}, line %d, index {index_printf}. Symbols: {symbols_printf}\\n", __LINE__, {index_expr}, {symbols_expr});
                exit(1);
            }}
            """,
            language=dace.Language.CPP,
            outputs={
                "__out": dace.Memlet.simple(newnode.data, index_expr, num_accesses=-1)
            },
            input_nodes={node.data: node},
            output_nodes={newnode.data: newnode},
            external_edges=True,
            # schedule=ScheduleType.GPU_Device,
        )
    print(f"Added {len(checks)} NaN checks")
