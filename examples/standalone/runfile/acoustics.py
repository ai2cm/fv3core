#!/usr/bin/env python3
import cProfile
import io
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
from fv3core.decorators import computepath_function
from fv3core.stencils.dyn_core import AcousticDynamics
from fv3core.utils.global_config import get_dacemode, set_dacemode
from fv3core.utils.grid import Grid
from fv3core.utils.null_comm import NullComm


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


def get_state_from_input(
    grid: Grid, input_data: Dict[str, Any]
) -> Dict[str, SimpleNamespace]:
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
    return {"state": statevars}


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


def run(data_directory, halo_update, backend, time_steps, reference_run):
    set_up_namelist(data_directory)
    serializer = initialize_serializer(data_directory)
    initialize_fv3core(backend, halo_update)
    grid = read_grid(serializer)
    spec.set_grid(grid)

    input_data = read_input_data(grid, serializer)

    state = get_state_from_input(grid, input_data)

    acoustics_object = AcousticDynamics(
        None,
        spec.namelist,
        input_data["ak"],
        input_data["bk"],
        input_data["pfull"],
        input_data["phis"],
    )
    state.__dict__.update(acoustics_object._temporaries)

    @computepath_function
    def iterate(state: dace.constant, time_steps):
        # @Linus: make this call a dace program
        for _ in range(time_steps):
            acoustics_object(state, insert_temporaries=False)

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
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            set_dacemode(dacemode)
        print(f"{backend} time:", time.time() - start)
    else:
        pr = cProfile.Profile()
        pr.enable()

        try:
            iterate(state, time_steps)
        finally:
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

    return state


@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("time_steps", required=False, default="1", type=int)
@click.argument("backend", required=False, default="gtc:gt:cpu_ifirst")
@click.option("--disable_halo_exchange/--no-disable_halo_exchange", default=False)
@click.option("--print_timings/--no-print_timings", default=True)
def driver(
    data_directory: str,
    time_steps: str,
    backend: str,
    disable_halo_exchange: bool,
    print_timings: bool,
):
    state = run(
        data_directory,
        not disable_halo_exchange,
        time_steps=time_steps,
        backend=backend,
        reference_run=not get_dacemode(),
    )
    ref_state = run(
        data_directory,
        not disable_halo_exchange,
        time_steps=time_steps,
        backend="numpy",
        reference_run=True,
    )

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
