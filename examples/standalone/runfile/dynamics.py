#!/usr/bin/env python3

import copy
import cProfile
import io
import json
import os
import pstats
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List

import click
import dace
import numpy as np
import serialbox


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


# Dev note: the GTC toolchain fails if xarray is imported after gt4py
# fv3gfs.util imports xarray if it's available in the env.
# fv3core imports gt4py.
# To avoid future conflict creeping back we make util imported prior to
# fv3core. isort turned off to keep it that way.
# isort: off
import fv3gfs.util as util
from fv3core.decorators import computepath_function
from fv3core.utils.global_config import set_dacemode, get_dacemode
from fv3core.utils.null_comm import NullComm

# isort: on

import fv3core
import fv3core._config as spec
import fv3core.testing


def set_experiment_info(
    experiment_name: str, time_step: int, backend: str, git_hash: str
) -> Dict[str, Any]:
    experiment: Dict[str, Any] = {}
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    experiment["setup"] = {}
    experiment["setup"]["timestamp"] = dt_string
    experiment["setup"]["dataset"] = experiment_name
    experiment["setup"]["timesteps"] = time_step
    experiment["setup"]["hash"] = git_hash
    experiment["setup"]["version"] = "python/" + backend
    experiment["setup"]["format_version"] = 2
    experiment["times"] = {}
    return experiment


def collect_keys_from_data(times_per_step: List[Dict[str, float]]) -> List[str]:
    """Collects all the keys in the list of dics and returns a sorted version"""
    keys = set()
    for data_point in times_per_step:
        for k, _ in data_point.items():
            keys.add(k)
    sorted_keys = list(keys)
    sorted_keys.sort()
    return sorted_keys


def gather_timing_data(
    times_per_step: List[Dict[str, float]],
    results: Dict[str, Any],
    comm: MPI.Comm,
    root: int = 0,
) -> Dict[str, Any]:
    """returns an updated version of  the results dictionary owned
    by the root node to hold data on the substeps as well as the main loop timers"""
    is_root = comm.Get_rank() == root
    keys = collect_keys_from_data(times_per_step)
    data: List[float] = []
    for timer_name in keys:
        data.clear()
        for data_point in times_per_step:
            if timer_name in data_point:
                data.append(data_point[timer_name])

        sendbuf = np.array(data)
        recvbuf = None
        if is_root:
            recvbuf = np.array([data] * comm.Get_size())
        comm.Gather(sendbuf, recvbuf, root=0)
        if is_root:
            results["times"][timer_name]["times"] = copy.deepcopy(recvbuf.tolist())
    return results


def write_global_timings(experiment: Dict[str, Any]) -> None:
    now = datetime.now()
    filename = now.strftime("%Y-%m-%d-%H-%M-%S")
    with open(filename + ".json", "w") as outfile:
        json.dump(experiment, outfile, sort_keys=True, indent=4)


def gather_hit_counts(
    hits_per_step: List[Dict[str, int]], results: Dict[str, Any]
) -> Dict[str, Any]:
    """collects the hit count across all timers called in a program execution"""
    for data_point in hits_per_step:
        for name, value in data_point.items():
            if name not in results["times"]:
                print(name)
                results["times"][name] = {"hits": value, "times": []}
            else:
                results["times"][name]["hits"] += value
    return results


def collect_data_and_write_to_file(
    args: SimpleNamespace,
    comm: MPI.Comm,
    hits_per_step,
    times_per_step,
    experiment_name,
) -> None:
    """
    collect the gathered data from all the ranks onto rank 0 and write the timing file
    """
    is_root = comm.Get_rank() == 0
    results = None
    if is_root:
        print("Gathering Times")
        results = set_experiment_info(
            experiment_name, args.time_step, args.backend, args.hash
        )
        results = gather_hit_counts(hits_per_step, results)

    results = gather_timing_data(times_per_step, results, comm)

    if is_root:
        write_global_timings(results)


@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("time_steps", required=False, default="1", type=int)
@click.argument("backend", required=False, default="gtc:gt:cpu_ifirst")
@click.argument("hash", required=False, default="")
@click.option("--disable_halo_exchange/--no-disable_halo_exchange", default=False)
@click.option("--disable_json_dump/--no-disable_json_dump", default=False)
@click.option("--print_timings/--no-print_timings", default=True)
@click.option("--profile/--no-profile", default=False)
def driver(
    data_directory: str,
    time_steps: str,
    backend: str,
    hash: str,
    disable_halo_exchange: bool,
    disable_json_dump: bool,
    print_timings: bool,
    profile: bool,
):
    timer = util.Timer()
    timer.start("total")
    with timer.clock("initialization"):
        args = SimpleNamespace(
            data_dir=data_directory,
            time_step=int(time_steps),
            backend=backend,
            hash=hash,
            disable_halo_exchange=disable_halo_exchange,
            disable_json_dump=disable_json_dump,
            print_timings=print_timings,
            profile=profile,
        )
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        profiler = None

        fv3core.set_backend(args.backend)
        fv3core.set_rebuild(False)
        fv3core.set_validate_args(False)

        spec.set_namelist(args.data_dir + "/input.nml")

        experiment_name = os.path.basename(os.path.normpath(args.data_dir))

        # set up of helper structures
        serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read,
            args.data_dir,
            "Generator_rank" + str(rank),
        )
        if args.disable_halo_exchange:
            mpi_comm = NullComm(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())
        else:
            mpi_comm = MPI.COMM_WORLD

        # get grid from serialized data
        grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
        grid_data = {}
        grid_fields = serializer.fields_at_savepoint(grid_savepoint)
        for field in grid_fields:
            grid_data[field] = serializer.read(field, grid_savepoint)
            if len(grid_data[field].flatten()) == 1:
                grid_data[field] = grid_data[field][0]
        grid = fv3core.testing.TranslateGrid(grid_data, rank).python_grid()
        spec.set_grid(grid)

        # set up grid-dependent helper structures
        layout = spec.namelist.layout
        partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
        communicator = util.CubedSphereCommunicator(mpi_comm, partitioner)

        # create a state from serialized data
        savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
        driver_object = fv3core.testing.TranslateFVDynamics([grid])
        input_data = driver_object.collect_input_data(serializer, savepoint_in)
        input_data["comm"] = communicator
        state = driver_object.state_from_inputs(input_data)

        dycore = fv3core.DynamicalCore(
            comm=communicator,
            grid_data=spec.grid.grid_data,
            stencil_factory=spec.grid.stencil_factory,
            damping_coefficients=spec.grid.damping_coefficients,
            config=spec.namelist.dynamical_core,
            ak=state["atmosphere_hybrid_a_coordinate"],
            bk=state["atmosphere_hybrid_b_coordinate"],
            phis=state["surface_geopotential"],
        )

    @computepath_function
    def iterate(state: dace.constant, time_steps: int):
        # @Linus: make this call a dace program
        for _ in range(time_steps):
            dycore.step_dynamics(
                state,
                input_data["consv_te"],
                input_data["do_adiabatic_init"],
                input_data["bdt"],
                input_data["ptop"],
                input_data["n_split"],
                input_data["ks"],
            )

    reference_run = False
    if reference_run:
        dacemode = get_dacemode()
        set_dacemode(False)
        import time

        start = time.time()
        pr = cProfile.Profile()
        pr.enable()

        try:
            iterate(state, time_steps)
        finally:
            pr.disable()
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            set_dacemode(dacemode)
        print(f"{backend} time:", time.time() - start)
    else:
        if args.profile:
            profiler = cProfile.Profile()
            profiler.enable()

        try:
            iterate(state, time_steps)
        finally:
            if args.profile:
                profiler.disable()
                s = io.StringIO()
                sortby = pstats.SortKey.CUMULATIVE
                ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
                ps.print_stats()
                print(s.getvalue())
                profiler.dump_stats(
                    f"fv3core_{experiment_name}_{args.backend}_{rank}.prof"
                )

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print("SUCCESS")


if __name__ == "__main__":
    driver()
