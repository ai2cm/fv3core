from argparse import ArgumentParser

from mpi4py import MPI
import serialbox
import yaml
import git
from datetime import datetime
import numpy as np
import json
import pathlib
from statistics import mean, median
from typing import List

import fv3core
import fv3core._config as spec
import fv3core.stencils.fv_dynamics as fv_dynamics
import fv3core.testing
import fv3gfs.util as util
from fv3core.testing import write_to_json
from fv3gfs.util import Timer

def print_and_write_global_timings(timer, experiment_name, time_step, backend, comm, root=0):
    is_root = comm.Get_rank() == root
    recvbuf = np.array(0.0)
    experiment = {}
    now = datetime.now()
    sha = git.Repo(
        pathlib.Path(__file__).parent.absolute(), search_parent_directories=True
    ).head.object.hexsha
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    filename = now.strftime("%Y-%m-%d-%H-%M-%S")
    experiment["setup"] = {}
    experiment["setup"]["experiment time"] = dt_string
    experiment["setup"]["data set"] = experiment_name
    experiment["setup"]["timesteps"] = time_step
    experiment["setup"]["hash"] = sha
    experiment["setup"]["version"] = "python/" + backend
    experiment["times"] = {}
    for name, value in timer.times.items():
        if is_root:
            print(name)
            experiment["times"][name] = {}
        for label, op in [("minimum", MPI.MIN), ("maximum", MPI.MAX), ("mean", MPI.SUM)]:
            comm.Reduce(np.array(value), recvbuf, op=op)
            if is_root:
                print(comm.Get_size())
                if label == "mean":
                    recvbuf /= comm.Get_size()
                print(f"    {label}: {recvbuf}")
                experiment["times"][name][label] = float(recvbuf)

    if is_root:       
        with open(filename + ".json", "w") as outfile:
            json.dump(experiment, outfile, sort_keys=True, indent=4)

if __name__ == "__main__":
    timer = Timer()
    timer.start("total")
    with timer.clock("init"):

        usage = "usage: python %(prog)s <data_dir> <timesteps> <backend>"
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
        args = parser.parse_args()
        backend = args.backend
        data_dir = args.data_dir
        time_step = args.time_step

        # # MPI stuff
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # fv3core specific setup
        fv3core.set_backend(backend)
        fv3core.set_rebuild(False)

        # namelist setup
        spec.set_namelist(data_dir + "/input.nml")

        nml2 = yaml.safe_load(
            open(
                data_dir + "/input.yml",
                "r",
            )
        )["namelist"]
        experiment_name = yaml.safe_load(
            open(
                data_dir + "/input.yml",
                "r",
            )
        )["experiment_name"]

        # set up of helper structures
        serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read,
            data_dir,
            "Generator_rank" + str(rank),
        )
        cube_comm = util.CubedSphereCommunicator(
            comm, util.CubedSpherePartitioner.from_namelist(nml2)
        )

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
        communicator = util.CubedSphereCommunicator(comm, partitioner)

        # create a state from serialized data
        savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
        driver_object = fv3core.testing.TranslateFVDynamics([grid])
        input_data = driver_object.collect_input_data(serializer, savepoint_in)
        input_data["comm"] = communicator
        state = driver_object.state_from_inputs(input_data)

        # warm-up timestep
        fv_dynamics.fv_dynamics(
            state,
            communicator,
            input_data["consv_te"],
            input_data["do_adiabatic_init"],
            input_data["bdt"],
            input_data["ptop"],
            input_data["n_split"],
            input_data["ks"],
        )

    with timer.clock("main"):
        # Run the dynamics
        for i in range(time_step - 1):
            fv_dynamics.fv_dynamics(
                state,
                communicator,
                input_data["consv_te"],
                input_data["do_adiabatic_init"],
                input_data["bdt"],
                input_data["ptop"],
                input_data["n_split"],
                input_data["ks"],
                timer
            )

    timer.stop("total")
    # collect times and output simple statistics
    print("Gathering Times")
    print_and_write_global_timings(timer, experiment_name, time_step, backend, comm)
