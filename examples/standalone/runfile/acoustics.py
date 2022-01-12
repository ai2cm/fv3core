#!/usr/bin/env python3
import cProfile
import io
from logging import warn
import pstats
from pstats import SortKey
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import click
import dace
import numpy as np
import serialbox
import yaml

import fv3core
import fv3core._config as spec
import fv3core.testing
import fv3gfs.util as util
from fv3core.stencils.dyn_core import AcousticDynamics
from fv3core.utils.global_config import get_dacemode, set_dacemode
from fv3core.utils.grid import Grid
from fv3core.utils.null_comm import NullComm
from fv3core.utils.stencil import computepath_function
import fv3gfs.util as fv3util


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def set_up_namelist(data_directory: str) -> None:
    """
    Reads the namelist at the given directory and sets
    the global fv3core config to it
    """
    spec.set_namelist(data_directory + "/input.nml")


def initialize_serializer(data_directory: str, rank: int = 0) -> serialbox.Serializer:
    """Creates a Serializer based on the data-directory and the rank"""
    return serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        data_directory,
        "Generator_rank" + str(rank),
    )


def read_grid(serializer: serialbox.Serializer, rank: int = 0) -> Grid:
    """Uses the serializer to generate a Grid object from serialized data"""
    grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = serializer.read(field, grid_savepoint)
        if len(grid_data[field].flatten()) == 1:
            grid_data[field] = grid_data[field][0]
    return fv3core.testing.TranslateGrid(grid_data, rank).python_grid()


def initialize_fv3core(backend: str, disable_halo_exchange: bool) -> None:
    """
    Initializes globalfv3core config to the arguments for single runs
    with the given backend and choice of halo updates
    """
    fv3core.set_backend(backend)
    fv3core.set_rebuild(False)
    fv3core.set_validate_args(False)


def read_input_data(grid: Grid, serializer: serialbox.Serializer) -> Dict[str, Any]:
    """Uses the serializer to read the input data from disk"""
    driver_object = fv3core.testing.TranslateDynCore([grid])
    savepoint_in = serializer.get_savepoint("DynCore-In")[0]
    return driver_object.collect_input_data(serializer, savepoint_in)


def get_state_from_input(grid: Grid, input_data: Dict[str, Any]):
    """
    Transforms the input data from the dictionary of strings
    to arrays into a state  we can pass in

    Input is a dict of arrays. These are transformed into Storage arrays
    useable in GT4Py

    This will also take care of reshaping the arrays into same sized
    fields as required by the acoustics
    """
    driver_object = fv3core.testing.TranslateDynCore([grid])
    driver_object._base.make_storage_data_input_vars(input_data)

    inputs = driver_object.inputs
    for name, properties in inputs.items():
        grid.quantity_dict_update(
            input_data, name, dims=properties["dims"], units=properties["units"]
        )

    statevars = SimpleNamespace(**input_data)
    return statevars


def set_up_communicator(
    disable_halo_exchange: bool,
) -> Tuple[Optional[MPI.Comm], Optional[util.CubedSphereCommunicator]]:
    layout = spec.namelist.layout
    partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
    if MPI is not None:
        comm = MPI.COMM_WORLD
    else:
        comm = None
    if not disable_halo_exchange:
        assert comm is not None
        cube_comm = util.CubedSphereCommunicator(comm, partitioner)
    else:
        cube_comm = util.CubedSphereCommunicator(NullComm(0, 0), partitioner)
    return comm, cube_comm


def get_experiment_name(
    data_directory: str,
) -> str:
    return yaml.safe_load(
        open(
            data_directory + "/input.yml",
            "r",
        )
    )["experiment_name"]


def initialize_timers() -> Tuple[util.Timer, util.Timer, List, List]:
    total_timer = util.Timer()
    total_timer.start("total")
    timestep_timer = util.Timer()
    return total_timer, timestep_timer, [], []


def read_and_reset_timer(timestep_timer, times_per_step, hits_per_step):
    times_per_step.append(timestep_timer.times)
    hits_per_step.append(timestep_timer.hits)
    timestep_timer.reset()
    return times_per_step, hits_per_step


def run(data_directory, halo_update, backend, time_steps, sdfg_path=None):
    print(f"Running {backend}")

    # Read grid & build state from input_data read from savepoint
    set_up_namelist(data_directory)
    serializer = initialize_serializer(data_directory)
    initialize_fv3core(backend, halo_update)
    grid = read_grid(serializer)
    spec.set_grid(grid)
    input_data = read_input_data(grid, serializer)
    state = get_state_from_input(grid, input_data)

    # Network communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cube_comm = fv3util.CubedSphereCommunicator(
        comm,
        fv3util.CubedSpherePartitioner(fv3util.TilePartitioner(spec.namelist.layout)),
    )

    # Init acoustics
    acoustics_dynamics = AcousticDynamics(
        comm=cube_comm,
        stencil_factory=spec.grid.stencil_factory,
        grid_data=grid,
        damping_coefficients=spec.grid.damping_coefficients,
        grid_type=spec.namelist.dynamical_core.grid_type,
        nested=False,
        stretched_grid=False,
        config=spec.namelist.dynamical_core.acoustic_dynamics,
        ak=input_data["ak"],
        bk=input_data["bk"],
        pfull=input_data["pfull"],
        phis=input_data["phis"],
        state=state,
    )
    state.__dict__.update(acoustics_dynamics._temporaries)

    # Build SDFG_PATH if option given and specialize for the right backend
    if sdfg_path != "":
        loop_name = "acoustics_loop_on_cpu"  # gtc:dace
        if backend == "gtc:dace:gpu":
            loop_name = "acoustics_loop_on_gpu"
        rank_str = ""
        if MPI.COMM_WORLD.Get_size() > 1:
            rank_str = str(rank)
        sdfg_path = f"{sdfg_path}{rank_str}/dacecache/{loop_name}"
    else:
        sdfg_path = None

    # Non orchestrated loop for all backends
    def acoustics_loop_non_orchestrated(state, time_steps):
        for _ in range(time_steps):
            acoustics_dynamics(state, update_temporaries=False)

    # CPU backend with orchestration (same code as GPU, but named different for
    # caching purposed)
    @computepath_function(load_sdfg=sdfg_path)
    def acoustics_loop_on_cpu(
        state: dace.constant,
        time_steps,
    ):
        for _ in range(time_steps):
            acoustics_dynamics(state, update_temporaries=False)

    # GPU backend with orchestration (same code as GPU, but named different for
    # caching purposed)
    @computepath_function(load_sdfg=sdfg_path)
    def acoustics_loop_on_gpu(
        state: dace.constant,
        time_steps,
    ):
        for _ in range(time_steps):
            acoustics_dynamics(state, update_temporaries=False)

    # Cache warm up and loop function selection
    if time_steps == 0:
        if backend == "gtc:dace":
            acoustics_loop_on_cpu(state, 1)
        elif backend == "gtc:dace:gpu":
            acoustics_loop_on_gpu(state, 1)
        else:
            dacemode = get_dacemode()
            set_dacemode(False)
            acoustics_loop_non_orchestrated(state, 1)
            set_dacemode(dacemode)
        print("Cached built - no loop run")
    else:
        if sdfg_path == None:
            warn(
                f"Running loop {time_steps} times but not SDFG was"
                f"given, performance will be poor."
            )

        # Get Rank
        rank = comm.Get_rank()

        # Simulate
        import time

        start = time.time()
        if backend == "gtc:dace":
            acoustics_loop_on_cpu(state, time_steps)
        elif backend == "gtc:dace:gpu":
            acoustics_loop_on_gpu(state, time_steps)
        else:
            dacemode = get_dacemode()
            set_dacemode(False)
            acoustics_loop_non_orchestrated(state, time_steps)
            set_dacemode(dacemode)

        elapsed = time.time() - start
        per_timestep = elapsed / (time_steps if time_steps != 0 else 1)
        print(
            f"Total {backend} time on rank {rank} for {time_steps} steps: "
            f"{elapsed}s ({per_timestep}s /timestep)"
        )

    return state


@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("time_steps", required=False, default="1", type=int)
@click.argument("backend", required=False, default="gtc:gt:cpu_ifirst")
@click.argument("sdfg_path", required=False, default="")
@click.option("--halo_update/--no-halo_update", default=False)
@click.option("--check_against_numpy/--no-check_against_numpy", default=False)
def driver(
    data_directory: str,
    time_steps: str,
    backend: str,
    sdfg_path: str,
    halo_update: bool,
    check_against_numpy: bool,
):
    state = run(
        data_directory,
        halo_update,
        time_steps=time_steps,
        backend=backend,
        sdfg_path=sdfg_path,
    )
    if check_against_numpy:
        ref_state = run(
            data_directory,
            halo_update,
            time_steps=time_steps,
            backend="numpy",
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
