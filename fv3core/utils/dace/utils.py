import time

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

from fv3core.utils.global_config import get_dacemode, DaCeOrchestration

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


# Rough timer & log for major operations of DaCe
class DaCeProgress:
    _mode = str(get_dacemode())

    def __init__(self, label):
        self.label = label

    @classmethod
    def log(cls, message: str):
        print(f"[{cls._mode}] {message}")

    def __enter__(self):
        DaCeProgress.log(f"{self.label}...")
        self.start = time.time()

    def __exit__(self, _type, _val, _traceback):
        elapsed = time.time() - self.start
        DaCeProgress.log(f"{self.label}...{elapsed}s.")
