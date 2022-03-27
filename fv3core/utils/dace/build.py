import os.path
from typing import Any, Callable, Optional, Tuple
from unicodedata import decomposition

import yaml
from numpy import partition

from fv3core.utils import global_config
from fv3core.utils.global_config import DaCeOrchestration, get_dacemode
from fv3core.utils.mpi import MPI


################################################
# Distributed compilation


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
        if int(size / 6) == 0:
            is_compiling = True
        elif rank % int(size / 6) == rank:
            is_compiling = True
        MPI = None
        # We need to set MPI to none again to turn off distributed compilation
    return is_compiling, comm


def unblock_waiting_tiles(comm, sdfg_path: str) -> None:
    if comm.Get_size() > 1:
        for tile in range(1, 6):
            tilesize = comm.Get_size() / 6
            comm.send(sdfg_path, dest=tile * tilesize + comm.Get_rank())


def top_tile_rank_from_decomposition_string(string):
    partitioner = global_config.get_partitioner()
    tilesize = partitioner.total_ranks / 6
    if tilesize == 1:
        return 0
    for rank in range(partitioner.total_ranks):
        if (
            string == "00"
            and partitioner.tile.on_tile_bottom(rank)
            and partitioner.tile.on_tile_left(rank)
        ):
            return rank % tilesize
        if (
            string == "01"
            and partitioner.tile.on_tile_bottom(rank)
            and not partitioner.tile.on_tile_left(rank)
            and not partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize
        if (
            string == "02"
            and partitioner.tile.on_tile_bottom(rank)
            and partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize
        if (
            string == "10"
            and not partitioner.tile.on_tile_bottom(rank)
            and not partitioner.tile.on_tile_top(rank)
            and partitioner.tile.on_tile_left(rank)
        ):
            return rank % tilesize
        if (
            string == "11"
            and not partitioner.tile.on_tile_bottom(rank)
            and not partitioner.tile.on_tile_top(rank)
            and not partitioner.tile.on_tile_left(rank)
            and not partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize
        if (
            string == "12"
            and not partitioner.tile.on_tile_bottom(rank)
            and not partitioner.tile.on_tile_top(rank)
            and partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize
        if (
            string == "20"
            and partitioner.tile.on_tile_top(rank)
            and partitioner.tile.on_tile_left(rank)
        ):
            return rank % tilesize
        if (
            string == "21"
            and partitioner.tile.on_tile_top(rank)
            and not partitioner.tile.on_tile_left(rank)
            and not partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize
        if (
            string == "22"
            and partitioner.tile.on_tile_top(rank)
            and partitioner.tile.on_tile_right(rank)
        ):
            return rank % tilesize

    return None


def top_tile_rank_to_decomposition_string(rank):
    partitioner = global_config.get_partitioner()
    if partitioner.tile.on_tile_bottom(rank):
        if partitioner.tile.on_tile_left(rank):
            return "00"
        if partitioner.tile.on_tile_right(rank):
            return "20"
        else:
            return "10"
    if partitioner.tile.on_tile_top(rank):
        if partitioner.tile.on_tile_left(rank):
            return "02"
        if partitioner.tile.on_tile_right(rank):
            return "22"
        else:
            return "12"
    else:
        if partitioner.tile.on_tile_left(rank):
            return "01"
        if partitioner.tile.on_tile_right(rank):
            return "21"
        else:
            return "11"


def read_target_rank(rank, filename=None):
    partitioner = global_config.get_partitioner()
    top_tile_rank = top_tile_equivalent(rank, partitioner.total_ranks)
    with open(filename) as decomposition:
        parsed_file = yaml.safe_load(decomposition)
        return parsed_file[top_tile_rank_to_decomposition_string(top_tile_rank)]


def top_tile_equivalent(rank, size):
    tilesize = size / 6
    return rank % tilesize


################################################

################################################
# SDFG load (both .sdfg file and build directory containing .so)

""" The below helpers use a dirty "once" global flag to allow for reentry many
    calls as those are called from function in a recursive pattern.
"""

_loaded_sdfg_once = False


def write_decomposition():
    from gt4py import config as gt_config

    path = f"{gt_config.cache_settings['root_path']}/.layout/"
    config_path = path + "decomposition.yml"
    os.makedirs(path, exist_ok=True)
    decomposition = {}
    for string in ["00", "01", "02", "10", "11", "12", "20", "21", "22"]:
        target_rank = top_tile_rank_from_decomposition_string(string)
        if target_rank is not None:
            decomposition.setdefault(string, target_rank)

    with open(config_path, "w") as outfile:
        yaml.dump(decomposition, outfile)


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

    # Guarding against bad usage of this function
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

    comm = MPI.COMM_WORLD
    config_path = f"{gt_config.cache_settings['root_path']}/.layout/decomposition.yml"
    if comm.Get_size() > 1:
        rank_str = f"_{read_target_rank(comm.Get_rank(), config_path):06d}"
    else:
        rank_str = ""

    sdfg_dir_path = f"{gt_config.cache_settings['root_path']}/.gt_cache{rank_str}/dacecache/{program_name}"
    if not os.path.isdir(sdfg_dir_path):
        raise RuntimeError(f"Precompiled SDFG is missing at {sdfg_dir_path}")

    print(
        f"[DaCe Config] Rank {MPI.COMM_WORLD.Get_rank()} loading SDFG {sdfg_dir_path}"
    )

    return sdfg_dir_path
