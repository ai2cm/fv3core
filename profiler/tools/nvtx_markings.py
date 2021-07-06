try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


def get_stencil_name(frame, event, args) -> str:
    """Get the name of the stencil from within a call to FrozenStencil.__call__"""
    name = getattr(
        frame.f_locals["self"].stencil_object,
        "__name__",
        repr(frame.f_locals["self"].stencil_object.options["name"]),
    )
    return f"{name}.__call__"


def get_name_from_frame(frame, event, args) -> str:
    """Static name from frame object"""
    return frame.f_code.co_name


""" List of hook descriptors

Each entry define a unique id (function name + filename[Optional]) and a function
that gives back a str for the marker.

TODO: this is a poor-person JSON, a ppjson if you will, it could be extracted as an
configuration file if there's a usage for it
"""
functions_desc = [
    {
        "fn": "__call__",
        "file": "fv3core/decorators.py",
        "name": get_stencil_name,
    },  # All call from StencilX decorators
    {
        "fn": "__call__",
        "file": "fv3core/stencils/dyn_core.py",
        "name": "Acoustic timestep",
    },
    {
        "fn": "__call__",
        "file": "fv3core/stencils/tracer_2d_1l.py",
        "name": "Tracer advection",
    },
    {"fn": "compute", "file": "fv3core/stencils/remapping.py", "name": "Remapping"},
    {
        "fn": "step_dynamics",
        "file": "fv3core/stencils/fv_dynamics.py",
        "name": get_name_from_frame,
    },
    {
        "fn": "halo_update",
        "file": None,
        "name": "HaloEx: sync scalar",
    },  # Synchroneous halo update
    {
        "fn": "vector_halo_update",
        "file": None,
        "name": "HaloEx: sync vector",
    },  # Synchroneous vector halo update
    {
        "fn": "start_halo_update",
        "file": None,
        "name": "HaloEx: async scalar",
    },  # Asynchroneous halo update
    {
        "fn": "start_vector_halo_update",
        "file": None,
        "name": "HaloEx: async vector",
    },  # Asynchroneous vector halo update
    {
        "fn": "wait",
        "file": "fv3gfs/util/communicator.py",
        "name": "HaloEx: unpack and wait",
    },  # Halo update finish
    {
        "fn": "_device_synchronize",
        "file": "fv3gfs/util/communicator.py",
        "name": "Pre HaloEx",
    },  # Synchronize all work prior to halo exchange
    {
        "fn": "_build_flatten_indices",
        "file": "fv3gfs/util/message.py",
        "name": "Build Idx",
    },
    {
        "fn": "async_pack",
        "file": "fv3gfs/util/message.py",
        "name": "async_pack",
    },
    {
        "fn": "async_unpack",
        "file": "fv3gfs/util/message.py",
        "name": "async_unpack",
    },
    {
        "fn": "synchronize",
        "file": "fv3gfs/util/message.py",
        "name": "synchronize",
    },
    {
        "fn": "finalize",
        "file": "fv3gfs/util/message.py",
        "name": "finalize",
    },
    {
        "fn": "allocate",
        "file": "fv3gfs/util/message.py",
        "name": "allocate",
    },
    {
        "fn": "_Isend_Irecv_halos",
        "file": "fv3gfs/util/communicator.py",
        "name": "_Isend_Irecv_halos",
    },
    {
        "fn": "async_exchange_start",
        "file": "fv3gfs/util/halo_updater.py",
        "name": "start async halo ex",
    },
    {
        "fn": "async_pack",
        "file": "fv3gfs/util/packed_buffer.py",
        "name": "pack to one buffer",
    },
    {
        "fn": "async_unpack",
        "file": "fv3gfs/util/packed_buffer.py",
        "name": "unpack to one buffer",
    },
    {
        "fn": "synchronize",
        "file": "fv3gfs/util/packed_buffer.py",
        "name": "internal sync",
    },
    {
        "fn": "async_exchange_wait",
        "file": "fv3gfs/util/halo_updater.py",
        "name": "stop async halo ex",
    },
    {
        "fn": "blocking_exchange",
        "file": "fv3gfs/util/halo_updater.py",
        "name": "blocking halo ex",
    },
    # Physics
    {
        "fn": "run",
        "file": "gfdl_cloud_microphys_gt4py.py",
        "name": "microph",
    },
    {
        "fn": "__call__",
        "file": "m_fields_init__gtcuda_98664b9438.py",
        "name": "fields_init",
    },
    {
        "fn": "__call__",
        "file": "m_icloud__gtcuda_fffa848465.py",
        "name": "warm_rain",
    },
    {
        "fn": "__call__",
        "file": "m_sedimentation__gtcuda_ad614ff181.py",
        "name": "sedimentation",
    },
    {
        "fn": "__call__",
        "file": "m_warm_rain__gtcuda_564217f553.py",
        "name": "icloud",
    },
    {
        "fn": "__call__",
        "file": "m_fields_update__gtcuda_7d4765ae41.py",
        "name": "fields_update",
    },
]


def mark(frame, event, args):
    """Hook at each function call & exit to record a Mark."""
    if event == "call":
        for fn_desc in functions_desc:
            if frame.f_code.co_name == fn_desc["fn"] and (
                fn_desc["file"] is None or fn_desc["file"] in frame.f_code.co_filename
            ):
                name = (
                    fn_desc["name"]
                    if isinstance(fn_desc["name"], str)
                    else fn_desc["name"](frame, event, args)
                )
                cp.cuda.nvtx.RangePush(name)
    elif event == "return":
        for fn_desc in functions_desc:
            if frame.f_code.co_name == fn_desc["fn"] and (
                fn_desc["file"] is None or fn_desc["file"] in frame.f_code.co_filename
            ):
                cp.cuda.nvtx.RangePop()
