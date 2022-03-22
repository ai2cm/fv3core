from typing import Callable, Tuple, Any, Optional

from fv3core.utils.mpi import MPI

from fv3core.utils.global_config import get_dacemode, DaCeOrchestration
import os.path

################################################
# Distributed compilation


def is_first_tile(rank: int, size: int) -> bool:
    return rank % int(size / 6) == rank


def determine_compiling_ranks() -> Tuple[bool, Any]:
    is_compiling = False
    rank = 0
    size = 1
    try:
        from fv3core.utils.mpi import MPI

        if MPI:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
    finally:
        if is_first_tile(rank, size):
            is_compiling = True
        MPI = None
        # We need to set MPI to none again to turn off distributed compilation
    return is_compiling, comm


def unblock_waiting_tiles(comm, sdfg_path: str) -> None:
    if comm.Get_size() > 1:
        for tile in range(1, 6):
            tilesize = comm.Get_size() / 6
            comm.send(sdfg_path, dest=tile * tilesize + comm.Get_rank())


def source_sdfg_rank(comm):
    tilesize = comm.Get_size() / 6
    my_rank = comm.Get_rank()
    return my_rank % tilesize


################################################

################################################
# SDFG load (both .sdfg file and build directory containing .so)

""" The below helpers use a dirty "once" global flag to allow for reentry many
    calls as those are called from function in a recursive pattern.
"""

_loaded_sdfg_once = False


def load_sdfg_once(
    program: Callable, sdfg_file_path: Optional[str] = None
) -> Optional[str]:
    """Attempt to load SDFG the first time it's called.
    Silently return for any other call but the first one.
    """

    if get_dacemode() != DaCeOrchestration.Run:
        return None

    global _loaded_sdfg_once
    if _loaded_sdfg_once:
        return None

    # Flag the function has called
    _loaded_sdfg_once = True

    # Qualified name as built by DaCe folder structure
    qualified_dirname = (
        f"{program.__module__}.{program.__qualname__}".replace(".", "_")
        .replace("__main__", "")
        .replace("_run_<locals>_", "")
    )

    return build_sdfg_path(qualified_dirname, sdfg_file_path)


def build_sdfg_path(program_name: str, sdfg_file_path: Optional[str] = None) -> str:
    """Build an SDFG path from the qualified program name or it's direct path to .sdfg

    Args:
        program_name: qualified name in the form module_qualname if module is not locals
        sdfg_file_path: absolute path to a .sdfg file
    """

    # Guarding agaisnt bad usage of this function
    if get_dacemode() != DaCeOrchestration.Run:
        raise RuntimeError(
            "Coding mistaked: sdfg path ask but DaCe orchestration is != Production"
        )

    # Case of a .sdfg file given by the user to be compiled
    if sdfg_file_path is not None:
        if not os.path.isfile(sdfg_file_path):
            raise RuntimeError(
                f"SDFG filepath {sdfg_file_path} cannot be found or is not a file"
            )
        return sdfg_file_path

    # Case of loading a precompiled .so - lookup using GT_CACHE
    import os
    from gt4py import config as gt_config

    if MPI.COMM_WORLD.Get_size() > 1:
        rank_str = f"_{int(source_sdfg_rank(MPI.COMM_WORLD)):06d}"
    else:
        rank_str = ""

    sdfg_dir_path = f"{gt_config.cache_settings['root_path']}/.gt_cache{rank_str}/dacecache/{program_name}"
    if not os.path.isdir(sdfg_dir_path):
        raise RuntimeError(f"Precompiled SDFG is missing at {sdfg_dir_path}")

    print(f"[DaCe Config] Rank {rank_str} loading SDFG {sdfg_dir_path}")

    return sdfg_dir_path
