"""Adding semantic marking to external profiler.

Usage: python external_profiler.py <PYTHON SCRIPT>.py <ARGS>

Works with nvtx (via cupy) for now.
"""

import fnmatch
import pickle
import sys
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from os import getcwd, getenv, listdir, mkdir, path, walk
from shutil import copy, copytree
from typing import Any, Dict, Tuple

import numpy as np
from gt4py import storage


try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


def parse_args():
    usage = "usage: python %(prog)s <--nvtx> <--stencil=STENCIL_NAME> <CMD TO PROFILE>"
    parser = ArgumentParser(usage=usage)

    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="enable NVTX marking",
    )
    parser.add_argument(
        "--stencil",
        type=str,
        action="store",
        help="create a small reproducer for the stencil",
    )
    return parser.parse_known_args()


def get_stencil_name(frame, event, args) -> str:
    """Get the name of the stencil from within a call to FrozenStencil.__call__"""
    name = getattr(
        frame.f_locals["self"].func, "__name__", repr(frame.f_locals["self"].func)
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
]


def nvtx_mark(frame, event, args):
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


STENCIL_CANDIDATE_FOR_EXTRACT: Dict[str, Tuple[str, str]] = {}
STENCIL_SERIALIZED_ARGUMENTS: Dict[str, Any] = {}  # Indexed on file path


def stencil_data_serialization(frame, event, args):
    if event == "call" or event == "return":
        for stencil_key, stencil_info in STENCIL_CANDIDATE_FOR_EXTRACT.items():
            if (
                frame.f_code.co_name == "run"
                and stencil_info[0] == frame.f_code.co_filename
            ):
                print(f"[PROFILER] Pickling args of {stencil_key} @ event {event}")
                if event == "call":
                    prefix = "pre_run_"
                else:
                    prefix = "post_run_"

                scalars = {}
                for arg_key, arg_value in frame.f_locals.items():
                    if arg_key == "self":
                        continue
                    if isinstance(arg_value, storage.Storage):
                        arg_value.device_to_host()
                        np.savez_compressed(
                            f"{stencil_info[1]}/data/{prefix}_{arg_key}.npz",
                            arg_value.data,
                        )
                    else:
                        scalars[arg_key] = arg_value
                scalar_file = f"{stencil_info[1]}/data/{prefix}_scalars.pickled"
                with open(scalar_file, "wb") as handle:
                    pickle.dump(scalars, handle, protocol=pickle.HIGHEST_PROTOCOL)


def find_stencil_candidate(stencil_name):
    expected_py_wrapper_partialname = f"m_{stencil_name}__"
    gt_cache_root = getenv("GT_CACHE_ROOT")
    gt_cache_root = gt_cache_root if gt_cache_root is not None else getcwd()
    print(f"[PROFILER] Searching for {stencil_name} in {gt_cache_root}...")
    for fname in listdir(gt_cache_root):
        fullpath = path.join(gt_cache_root, fname)
        if fname.startswith(".gt_cache") and path.isdir(fullpath):
            for root, _dirnames, filenames in walk(fullpath):
                for py_wrapper_file in fnmatch.filter(
                    filenames, f"{expected_py_wrapper_partialname}*.py"
                ):
                    print(f"...found candidate {path.join(root, py_wrapper_file)}")
                    stencil_key = path.splitext(py_wrapper_file)[0]
                    stencil_file_wrapper = path.join(root, py_wrapper_file)
                    STENCIL_CANDIDATE_FOR_EXTRACT[stencil_key] = (
                        stencil_file_wrapper,
                        None,
                    )
    if len(STENCIL_CANDIDATE_FOR_EXTRACT.items()) != 0:
        # Create the result dir
        repro_dir = (
            f"{getcwd()}/repro_{stencil_name}_"
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        mkdir(repro_dir)
        # Copy required file for repro & prepare for args pickling
        for stencil_key, stencil_info in STENCIL_CANDIDATE_FOR_EXTRACT.items():
            # One folder per stencil candidate
            stencil_dir = f"{repro_dir}/{stencil_key}"
            mkdir(stencil_dir)
            # Save info
            stencil_info = (stencil_info[0], stencil_dir)
            STENCIL_CANDIDATE_FOR_EXTRACT[stencil_key] = stencil_info
            # Create data directories
            mkdir(f"{stencil_dir}/data")
            # Copy original code
            origin_code_copy_dir = f"{stencil_dir}/original_code"
            mkdir(origin_code_copy_dir)
            widlcard = f"{path.dirname(stencil_info[0])}/{stencil_key[:-2]}*"
            for orignal_file in glob(widlcard):
                if path.isfile(orignal_file):
                    copy(orignal_file, origin_code_copy_dir)
                if path.isdir(orignal_file):
                    copytree(orignal_file, origin_code_copy_dir, dirs_exist_ok=True)
            # Write reproducer script
            with open(f"{stencil_dir}/repro.py", "w") as handle:
                handle.write(
                    f"""from original_code import {stencil_key}
from os import path
import pickle
from gt4py import storage
import cupy as cp
import numpy as np

if __name__ == "__main__":
    # Load compiled object
    root_dir = path.dirname(path.realpath(__file__))
    compute_object = (
        {stencil_key}.{stencil_key[2:].replace('__', '____')}()
    )
    # Select a module depending on backend to load the serialized data
    loading_module = np
    if compute_object._gt_backend_ == "gtcuda":
        loading_module = cp
    # Setup the fields
    arguments = {{}}
    for field_name, _field_info in compute_object._gt_field_info_.items():
        field_file = f"{{root_dir}}/data/pre_run__{{field_name}}.npz"
        with cp.load(field_file) as npz_handle:
            arguments[field_name] = storage.from_array(
                npz_handle["arr_0"], compute_object._gt_backend_, (0, 0, 0)
            )
    # Un-pickle the scalars and finalize the argument list
    with open(root_dir + "/data/pre_run__scalars.pickled", "rb") as handle:
        arguments.update(pickle.load(handle))
    compute_object.run(**arguments)
"""
                )


def extract_stencils():
    pass


def profile_hook(frame, event, args):
    if cmd_line_args.nvtx:
        nvtx_mark(frame, event, args)
    if cmd_line_args.stencil:
        stencil_data_serialization(frame, event, args)


cmd_line_args = None
if __name__ == "__main__":
    cmd_line_args, unknown = parse_args()
    print(f"{cmd_line_args}")
    print(f"{unknown}")
    if cmd_line_args.nvtx and cp is None:
        print("WARNING: cupy isn't available, NVTX marking deactivated.")
        cmd_line_args.nvtx = False
    if cmd_line_args.stencil is not None:
        find_stencil_candidate(cmd_line_args.stencil)
    if cmd_line_args.nvtx or cmd_line_args.stencil:
        sys.setprofile(profile_hook)
    filename = unknown[0]
    sys.argv = unknown[0:]
    exec(compile(open(filename, "rb").read(), filename, "exec"))
