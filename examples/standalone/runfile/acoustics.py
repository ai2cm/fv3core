
#!/usr/bin/env python3

import copy
import json
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Any, Dict, List
from types import SimpleNamespace

import numpy as np
import yaml
from mpi4py import MPI

import fv3core
import fv3core._config as spec
import fv3core.testing
import fv3core.utils.global_config as global_config
import fv3gfs.util as util
import serialbox
import timing

def parse_args() -> Namespace:
    usage = (
        "usage: python %(prog)s <data_dir> <timesteps> <backend> <hash> <halo_exchange>"
    )
    parser = ArgumentParser(usage=usage)

    parser.add_argument(
        "data_dir",
        type=str,
        action="store",
        help="directory containing data to run with",
    )
    parser.add_argument(
        "time_step",
        type=int,
        action="store",
        help="number of timesteps to execute",
    )
    parser.add_argument(
        "backend",
        type=str,
        action="store",
        help="path to the namelist",
    )
    parser.add_argument(
        "hash",
        type=str,
        action="store",
        help="git hash to store",
    )
    parser.add_argument(
        "--disable_halo_exchange",
        action="store_true",
        help="enable or disable the halo exchange",
    )
    parser.add_argument(
        "--disable_json_dump",
        action="store_true",
        help="enable or disable json dump",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="enable performance profiling using cProfile",
    )

    return parser.parse_args()


def read_grid( serializer: serialbox.Serializer, rank: int = 0) -> fv3core.testing.TranslateGrid:
    grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = serializer.read(field, grid_savepoint)
        if len(grid_data[field].flatten()) == 1:
            grid_data[field] = grid_data[field][0]
    grid = fv3core.testing.TranslateGrid(grid_data, rank).python_grid()
    return grid

def read_input_data(grid, serializer):
    savepoint_in = serializer.get_savepoint("DynCore-In")[0]
    driver_object = fv3core.testing.TranslateDynCore([grid])
    input_data = driver_object.collect_input_data(serializer, savepoint_in)
    driver_object._base.make_storage_data_input_vars(input_data)
    for name, properties in driver_object.inputs.items():
        driver_object.grid.quantity_dict_update(
            input_data, name, dims=properties["dims"], units=properties["units"]
        )
        
    return input_data


if __name__ == "__main__":
    timer = util.Timer()
    timer.start("total")
    with timer.clock("initialization"):
        args = parse_args()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        profiler = None
        if args.profile:
            import cProfile

            profiler = cProfile.Profile()
            profiler.disable()

        fv3core.set_backend(args.backend)
        fv3core.set_rebuild(False)
        fv3core.set_validate_args(False)
        global_config.set_do_halo_exchange(not args.disable_halo_exchange)

        spec.set_namelist(args.data_dir + "/input.nml")

        experiment_name = yaml.safe_load(
            open(
                args.data_dir + "/input.yml",
                "r",
            )
        )["experiment_name"]

        # set up of helper structures
        serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read,
            args.data_dir,
            "Generator_rank" + str(rank),
        )
        cube_comm = util.CubedSphereCommunicator(
            comm,
            util.CubedSpherePartitioner(util.TilePartitioner(spec.namelist.layout)),
        )

        # get grid from serialized data
        grid = read_grid(serializer, rank)
        spec.set_grid(grid)

        # set up grid-dependent helper structures
        layout = spec.namelist.layout
        partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
        communicator = util.CubedSphereCommunicator(comm, partitioner)

        # create a state from serialized data
        input_data = read_input_data(grid, serializer)
        state = SimpleNamespace(**input_data)

        acoustics = fv3core.AcousticDynamics(
            communicator,
            spec.namelist,
            state.ak,
            state.bk,
            state.pfull,
            state.phis
        )
        state.__dict__.update(acoustics._temporaries)

        # warm-up timestep.
        # We're intentionally not passing the timer here to exclude
        # warmup/compilation from the internal timers
        if rank == 0:
            print("timestep 1")
        acoustics(state,insert_temporaries=False)

    if profiler is not None:
        profiler.enable()

    times_per_step = []
    hits_per_step = []
    # we set up a specific timer for each timestep
    # that is cleared after so we get individual statistics
    timestep_timer = util.Timer()
    for i in range(args.time_step - 1):
        # this loop is not required, but make performance numbers comparable with FVDynamics
        for _ in range(spec.namelist.k_split):
            with timestep_timer.clock("DynCore"):
                if rank == 0:
                    print(f"timestep {i+2}")
                acoustics(state,insert_temporaries=False)
        times_per_step.append(timestep_timer.times)
        hits_per_step.append(timestep_timer.hits)
        timestep_timer.reset()

    if profiler is not None:
        profiler.disable()

    timer.stop("total")
    times_per_step.append(timer.times)
    hits_per_step.append(timer.hits)

    # output profiling data
    if profiler is not None:
        profiler.dump_stats(f"fv3core_{experiment_name}_{args.backend}_{rank}.prof")

    # Timings
    if not args.disable_json_dump:
        # Collect times and output statistics in json
        comm.Barrier()
        timing.collect_data_and_write_to_file(
            args, comm, hits_per_step, times_per_step, experiment_name
        )
    else:
        # Print a brief summary of timings
        # Dev Note: we especially do _not_ gather timings here to have a
        # no-MPI-communication codepath
        print(f"Rank {rank} done. Total time: {timer.times['total']}.")

    if rank == 0:
        print("SUCCESS")
