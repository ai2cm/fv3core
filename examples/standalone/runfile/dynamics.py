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
from fv3core.utils import global_config
import serialbox
from warnings import warn


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


try:
    import cupy as cp
except ImportError:
    cp = None

# Dev note: the GTC toolchain fails if xarray is imported after gt4py
# fv3gfs.util imports xarray if it's available in the env.
# fv3core imports gt4py.
# To avoid future conflict creeping back we make util imported prior to
# fv3core. isort turned off to keep it that way.
# isort: off
import fv3gfs.util as util
from fv3core.utils.global_config import set_dacemode, get_dacemode
from fv3core.utils.null_comm import NullComm

# isort: on

import fv3core
import fv3core._config as spec
import fv3core.testing

# [DaCe] `get_namespace`: Transform state outside of FV_Dynamics in order to
#        have valid references in halo ex callbacks
from fv3core.decorators import get_namespace
import fv3core.stencils.fv_dynamics as fv_dynamics
from fv3core.utils.dace.computepath import computepath_function


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


def run(
    data_directory: str,
    time_steps: str,
    backend: str,
    hash: str,
    sdfg_path: str,
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

        fv3core.set_backend(backend)
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
        dict_state = driver_object.state_from_inputs(input_data)
        # [DaCe] `get_namespace`: Transform state outside of FV_Dynamics in order to
        #        have valid references in halo ex callbacks
        state = get_namespace(fv_dynamics.DynamicalCoreArgSpec.values, dict_state)

        print(
            f"Config\n"
            f"\tBackend {backend}\n"
            f"\tOrchestration: {'dace' if get_dacemode() else 'python'}\n"
            f"\tN split: {spec.namelist.dynamical_core.n_split}\n"
            f"\tK split: {spec.namelist.dynamical_core.k_split}\n"
        )

        dycore = fv3core.DynamicalCore(
            comm=communicator,
            grid_data=spec.grid.grid_data,
            stencil_factory=spec.grid.stencil_factory,
            damping_coefficients=spec.grid.damping_coefficients,
            config=spec.namelist.dynamical_core,
            ak=dict_state["atmosphere_hybrid_a_coordinate"],
            bk=dict_state["atmosphere_hybrid_b_coordinate"],
            phis=dict_state["surface_geopotential"],
            state=state,
            timer=timer,
        )
        dycore.update_state(
            input_data["consv_te"],
            input_data["bdt"],
            input_data["do_adiabatic_init"],
            input_data["ptop"],
            input_data["n_split"],
            input_data["ks"],
            state,
        )

    @computepath_function
    def dycore_loop_on_cpu(state: dace.constant, time_steps: int):
        for _ in dace.nounroll(range(time_steps)):
            dycore.step_dynamics(
                state,
            )

    @computepath_function
    def dycore_loop_on_gpu(state: dace.constant, time_steps: int):
        for _ in dace.nounroll(range(time_steps)):
            dycore.step_dynamics(
                state,
            )

    @computepath_function
    def c_sw_loop_on_gpu(state: dace.constant, time_steps: int):
        for _ in dace.nounroll(range(time_steps)):
            # -- C_SW -- #
            dt = state.mdt / dycore.config.n_split
            dt2 = 0.5 * dt
            dycore.acoustic_dynamics.cgrid_shallow_water_lagrangian_dynamics(
                state.delp,
                state.pt,
                state.u,
                state.v,
                state.w,
                state.uc,
                state.vc,
                state.ua,
                state.va,
                state.ut,
                state.vt,
                state.divgd,
                state.omga,
                dt2,
            )

    @computepath_function
    def c_sw_loop_on_cpu(state: dace.constant, time_steps: int):
        for _ in dace.nounroll(range(time_steps)):
            # -- C_SW -- #
            dt = state.mdt / dycore.config.n_split
            dt2 = 0.5 * dt
            dycore.acoustic_dynamics.cgrid_shallow_water_lagrangian_dynamics(
                state.delp,
                state.pt,
                state.u,
                state.v,
                state.w,
                state.uc,
                state.vc,
                state.ua,
                state.va,
                state.ut,
                state.vt,
                state.divgd,
                state.omga,
                dt2,
            )

    def dycore_loop_non_orchestrated(state: dace.constant, time_steps: int):
        for _ in range(time_steps):
            dycore.step_dynamics(
                state,
            )

    # Cache warm up and loop function selection
    dace_orchestrated_backend = (
        "dace" in backend and global_config.is_dace_orchestrated()
    )
    print("Cache warming run")

    dycore_fn = None
    if dace_orchestrated_backend and backend == "gtc:dace":
        dycore_fn = dycore_loop_on_cpu
        # dycore_fn = c_sw_loop_on_gpu
    elif dace_orchestrated_backend and backend == "gtc:dace:gpu":
        dycore_fn = dycore_loop_on_gpu
        # dycore_fn = c_sw_loop_on_gpu
    else:
        dycore_fn = dycore_loop_non_orchestrated

    if not dace_orchestrated_backend:
        print("Running non-orchestrated")
        dacemode = get_dacemode()
        set_dacemode(False)
        dycore_fn(state, 1)
        set_dacemode(dacemode)
    else:
        dycore_fn(state, 1)

    # Ready to time
    timer.reset()

    if time_steps == 0:
        print("Cached built only - no benchmarked run")
        return

    if cp is not None:
        cp.cuda.nvtx.RangePush("Performance Run")

    if dace_orchestrated_backend:
        with timer.clock("mainloop_orchestrated"):
            dycore_fn(state, time_steps)
    else:
        with timer.clock("mainloop_not_orchestrated"):
            dycore_fn(state, time_steps)
        set_dacemode(dacemode)

    timer.stop("total")

    if cp is not None:
        cp.cuda.nvtx.RangePop()

    # Timings
    # Print a brief summary of timings
    # Dev Note: we especially do _not_ gather timings here to have a
    # no-MPI-communication codepath
    print(f"Rank {rank} done. Timer: {timer.times} {timer.hits}.")

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print("SUCCESS")

    return state


@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("time_steps", required=False, default="1", type=int)
@click.argument("backend", required=False, default="gtc:gt:cpu_ifirst")
@click.argument("hash", required=False, default="")
@click.argument("sdfg_path", required=False, default="")
@click.option("--disable_halo_exchange/--no-disable_halo_exchange", default=False)
@click.option("--disable_json_dump/--no-disable_json_dump", default=False)
@click.option("--print_timings/--no-print_timings", default=True)
@click.option("--profile/--no-profile", default=False)
@click.option("--check_against_numpy/--no-check_against_numpy", default=False)
def driver(
    data_directory: str,
    time_steps: str,
    backend: str,
    hash: str,
    sdfg_path: str,
    disable_halo_exchange: bool,
    disable_json_dump: bool,
    print_timings: bool,
    profile: bool,
    check_against_numpy: bool,
):
    state = run(
        data_directory=data_directory,
        time_steps=time_steps,
        backend=backend,
        hash=hash,
        sdfg_path=sdfg_path,
        disable_halo_exchange=disable_halo_exchange,
        disable_json_dump=disable_json_dump,
        print_timings=print_timings,
        profile=profile,
    )
    if check_against_numpy:
        ref_state = run(
            data_directory=data_directory,
            time_steps=time_steps,
            backend="numpy",
            hash=hash + "_numpy",
            sdfg_path=sdfg_path,
            disable_halo_exchange=disable_halo_exchange,
            disable_json_dump=disable_json_dump,
            print_timings=print_timings,
            profile=profile,
        )

    if check_against_numpy:
        for name, ref_value in ref_state.__dict__.items():

            if name in {"mfxd", "mfyd"}:
                continue
            value = state.__dict__[name]
            if isinstance(ref_value, util.quantity.Quantity):
                ref_value = ref_value.storage
            if isinstance(value, util.quantity.Quantity):
                value = value.storage
            if hasattr(value, "device_to_host"):
                value.device_to_host()
            if hasattr(value, "shape") and len(value.shape) == 3:
                value = np.asarray(value)[1:-1, 1:-1, :]
                ref_value = np.asarray(ref_value)[1:-1, 1:-1, :]
            np.testing.assert_allclose(ref_value, value, err_msg=name)


if __name__ == "__main__":
    driver()
