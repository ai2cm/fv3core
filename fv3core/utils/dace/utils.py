import time

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


# Callback to insert nvtx markings


def local_dace_inhibitor(f):
    return f


@local_dace_inhibitor
def cb_nvtx_range_push_dynsteps():
    if cp:
        cp.cuda.nvtx.RangePush("Dynamics.step")


@local_dace_inhibitor
def cb_nvtx_range_pop():
    if cp:
        cp.cuda.nvtx.RangePop()


# Rough timer for major operation of BUILD


class BuildProgress:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        print(f"[DaCe BUILD] {self.label}...")
        self.start = time.time()

    def __exit__(self, _type, _val, _traceback):
        elapsed = time.time() - self.start
        print(f"[DaCe BUILD] {self.label}...{elapsed}s.")


def get_sdfg_path(sdfg_path: str, loop_name: str) -> str:
    import os
    from mpi4py import MPI

    # Build SDFG_PATH if option given and specialize for the right backend
    return_sdfg_path = sdfg_path
    if not os.path.isfile(sdfg_path):
        if sdfg_path != "":
            rank_str = ""
            if MPI.COMM_WORLD.Get_size() > 1:
                rank_str = f"_00000{str(MPI.COMM_WORLD.Get_rank())}"
            return_sdfg_path = f"{sdfg_path}{rank_str}/dacecache/{loop_name}"
        else:
            return_sdfg_path = None

    print(f"Loading SDFG {return_sdfg_path}")
    return return_sdfg_path
