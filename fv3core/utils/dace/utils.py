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
