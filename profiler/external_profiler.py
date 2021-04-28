"""Adding semantic marking to external profiler

Works with nvtx (via cupy) for now.
"""

import sys

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None


def get_stencil_name(frame, event, args) -> str:
    """Get the name of the stencil from within a call to StencilWrapper.__call__"""
    name = getattr(
        frame.f_locals["self"].func, "__name__", repr(frame.f_locals["self"].func)
    )
    return f"{name}.__call__"


def get_static_name(frame, event, args) -> str:
    """Static naming"""
    if (
        frame.f_code.co_name == "__call__"
        and frame.f_code.co_filename == "fv3core/stencils/dyn_core.py"
    ):
        return "Acoustic timestep"
    elif (
        frame.f_code.co_name == "compute"
        and frame.f_code.co_filename == "fv3core/stencils/remapping.py"
    ):
        return "Remapping"


def get_name_from_frame(frame, event, args) -> str:
    """Static name from frame object"""
    return frame.f_code.co_name


""" List of hook descriptors

Each entry define a unique id (function name + filename[Optional]) and a function
that gives back a str for the marker.
"""
functions_desc = [
    {"fn": "__call__", "file": "fv3core/decorators.py", "name_fn": get_stencil_name},
    {
        "fn": "__call__",
        "file": "fv3core/stencils/dyn_core.py",
        "name_fn": get_static_name,
    },
    {
        "fn": "compute",
        "file": "fv3core/stencils/remapping.py",
        "name_fn": get_static_name,
    },
    {"fn": "fv_dynamics", "file": None, "name_fn": get_name_from_frame},
]


def profile_hook(frame, event, args):
    """Hook at each function call & exit to record a Mark"""
    if event == "call":
        for fn_desc in functions_desc:
            if frame.f_code.co_name == fn_desc["fn"] and (
                fn_desc["file"] is None or fn_desc["file"] in frame.f_code.co_filename
            ):
                name = fn_desc["name_fn"](frame, event, args)
                cp.cuda.nvtx.RangePush(name)
    elif event == "return":
        for fn_desc in functions_desc:
            if frame.f_code.co_name == fn_desc["fn"] and (
                fn_desc["file"] is None or fn_desc["file"] in frame.f_code.co_filename
            ):
                cp.cuda.nvtx.RangePop()


if __name__ == "__main__":
    if cp is None:
        raise RuntimeError("External profiling requires CUPY")
    sys.setprofile(profile_hook)
    filename = sys.argv[1]
    sys.argv = sys.argv[1:]
    exec(compile(open(filename, "rb").read(), filename, "exec"))
