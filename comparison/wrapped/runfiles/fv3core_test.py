import sys
import fv3gfs.wrapper as wrapper
import fv3core
from fv3gfs.util import Quantity, io, CubedSphereCommunicator, CubedSpherePartitioner
import numpy as np
import xarray as xr
import yaml
import mpi4py
import fv3core._config as spec

sys.path.append("/serialbox2/python")  # noqa: E402
sys.path.append("/fv3core/tests/translate")  # noqa: E402
import serialbox
import translate as translate

# May need to run 'ulimit -s unlimited' before running this example
# If you're running in our prepared docker container, you definitely need to do this
# sets the stack size to unlimited

# Run using mpirun -n 6 python3 basic_model.py
# mpirun flags that may be useful:
#     for docker:  --allow-run-as-root
#     for CircleCI: --oversubscribe
#     to silence a certain inconsequential MPI error: --mca btl_vader_single_copy_mechanism none

# All together:
# mpirun -n 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none python3 basic_model.py

# def init_tracers(shape):
#     arr = np.zeros(shape)
#     turbulent_kinetic_energy = Quantity.from_data_array(arr)

if __name__ == "__main__":

    # read in the namelist
    spec.set_namelist("input.nml")
    # nsplit = spec.namelist["n_split"]
    # consv_te = spec.namelist["consv_te"]
    dt_atmos = spec.namelist.dt_atmos

    # get another namelist for the communicator??
    nml2 = yaml.safe_load(open("/fv3core/comparison/wrapped/config/c12_6ranks_standard.yml", "r"))["namelist"]

    # set backend
    fv3core.utils.gt4py_utils.backend = "numpy"

    # MPI stuff
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    cube_comm = CubedSphereCommunicator(
        comm, CubedSpherePartitioner.from_namelist(nml2)
    )

    # Set the names of quantities in State
    names0 = [
        "specific_humidity",
        "cloud_water_mixing_ratio",
        "rain_mixing_ratio",
        "snow_mixing_ratio",
        "cloud_ice_mixing_ratio",
        "graupel_mixing_ratio",
        "ozone_mixing_ratio",
        "cloud_fraction",
        "air_temperature",
        "pressure_thickness_of_atmospheric_layer",
        "vertical_thickness_of_atmospheric_layer",
        "logarithm_of_interface_pressure",
        "x_wind",
        "y_wind",
        "vertical_wind",
        "x_wind_on_c_grid",
        "y_wind_on_c_grid",
        "total_condensate_mixing_ratio",
        "interface_pressure",
        "surface_geopotential",
        "interface_pressure_raised_to_power_of_kappa",
        "surface_pressure",
        "vertical_pressure_velocity",
        "atmosphere_hybrid_a_coordinate",
        "atmosphere_hybrid_b_coordinate",
        "accumulated_x_mass_flux",
        "accumulated_y_mass_flux",
        "accumulated_x_courant_number",
        "accumulated_y_courant_number",
        "dissipation_estimate_from_heat_source",
        "eastward_wind",
        "northward_wind",
        "layer_mean_pressure_raised_to_power_of_kappa",
        "time"
    ]

    names = [
        "specific_humidity",
        "cloud_water_mixing_ratio",
        "rain_mixing_ratio",
        "snow_mixing_ratio",
        "cloud_ice_mixing_ratio",
        "graupel_mixing_ratio",
        "ozone_mixing_ratio",
        "cloud_fraction",
        "turbulent_kinetic_energy",
        "air_temperature",
        "pressure_thickness_of_atmospheric_layer",
        "vertical_thickness_of_atmospheric_layer",
        "logarithm_of_interface_pressure",
        "x_wind",
        "y_wind",
        "vertical_wind",
        "eastward_wind",
        "northward_wind",
        "x_wind_on_c_grid",
        "y_wind_on_c_grid",
        "total_condensate_mixing_ratio",
        "interface_pressure",
        "surface_geopotential",
        "interface_pressure_raised_to_power_of_kappa",
        "layer_mean_pressure_raised_to_power_of_kappa",
        "surface_pressure",
        "vertical_pressure_velocity",
        "atmosphere_hybrid_a_coordinate",
        "atmosphere_hybrid_b_coordinate",
        "accumulated_x_mass_flux",
        "accumulated_y_mass_flux",
        "accumulated_x_courant_number",
        "accumulated_y_courant_number",
        "dissipation_estimate_from_heat_source",
        "time"
    ]

    # get grid from serialized data
    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        "/fv3core/test_data/c12_6ranks_standard",
        "Generator_rank" + str(rank),
    )
    grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = serializer.read(field, grid_savepoint)
        if len(grid_data[field].flatten()) == 1:
            grid_data[field] = grid_data[field][0]
    grid = translate.TranslateGrid(grid_data, rank).python_grid()
    fv3core._config.set_grid(grid)

    # startup
    wrapper.initialize()

    # add things to State
    origin = (0, 3, 3)
    extent = (spec.namelist.npz, spec.namelist.npy - 1, spec.namelist.npx - 1)
    arr = np.zeros(
        (spec.namelist.npz + 1, spec.namelist.npy + 6, spec.namelist.npx + 6)
    )
    turbulent_kinetic_energy = Quantity.from_data_array(
        xr.DataArray(
            arr,
            attrs={"fortran_name": "qsgs_tke", "units": "m**2/s**2"},
            dims=["z", "y", "x"],
        ),
        origin=origin,
        extent=extent,
    )

    flags = wrapper.Flags()
    for i in range(wrapper.get_step_count()):
        if i == 0:
            state = wrapper.get_state(names=names0)
            state["turbulent_kinetic_energy"] = turbulent_kinetic_energy
            # just gotta make sure everything is the right shape...
            # for quant in state.keys():
            #     state[quant] = state[quant].transpose([X_DIMS, Y_DIMS, Z_DIMS])
        else:
            state = wrapper.get_state(names=names)
        fv3core.fv_dynamics(
            state,
            cube_comm,
            flags.consv_te,
            flags.do_adiabatic_init,
            dt_atmos,
            flags.ptop,
            flags.n_split,
            flags.ks,
        )
        wrapper.set_state(state)
        wrapper.step_physics()
        wrapper.save_intermediate_restart_if_enabled()
    state = wrapper.get_state(names=names)
    # state["time"] = "{0}.{1}".format(spec.namelist["minutes"], spec.namelist["seconds"])
    print("HEY!")
    print(type(state["surface_pressure"]))
    io.write_state(state, "outstate_{0}.nc".format(rank))
    wrapper.cleanup()
