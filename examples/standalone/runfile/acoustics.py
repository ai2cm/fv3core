#!/usr/bin/env python3
from types import SimpleNamespace
from typing import Any, Dict

import click
import serialbox

import fv3core
import fv3core._config as spec
import fv3core.testing
import fv3core.utils.global_config as global_config
from fv3core.stencils.dyn_core import AcousticDynamics
from fv3core.utils.grid import Grid


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


def initialize_fv3core(backend: str, do_halo_updates: bool) -> None:
    """
    Initializes globalfv3core config to the arguments for single runs
    with the given backend and choice of halo updates
    """
    fv3core.set_backend(backend)
    fv3core.set_rebuild(False)
    fv3core.set_validate_args(False)
    global_config.set_do_halo_exchange(do_halo_updates)


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


@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("time_steps", required=False, default="1")
@click.argument("backend", required=False, default="gtc:gt:cpu_ifirst")
@click.option("--halo_update/--no-halo_update", default=False)
def driver(
    data_directory: str,
    time_steps: str,
    backend: str,
    halo_update: bool,
):
    set_up_namelist(data_directory)
    serializer = initialize_serializer(data_directory)
    initialize_fv3core(backend, halo_update)
    grid = read_grid(serializer)
    spec.set_grid(grid)

    input_data = read_input_data(grid, serializer)

    acoutstics_object = AcousticDynamics(
        None,
        spec.namelist,
        input_data["ak"],
        input_data["bk"],
        input_data["pfull"],
        input_data["phis"],
    )

    state = get_state_from_input(grid, input_data)

    for _ in range(int(time_steps)):
        acoutstics_object(**state)


if __name__ == "__main__":
    driver()
