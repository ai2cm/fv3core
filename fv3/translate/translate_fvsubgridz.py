from .parallel_translate import ParallelTranslateBaseSlicing
import fv3.stencils.fv_subgridz as fv_subgridz
import fv3util
import pytest
import fv3.utils.gt4py_utils as utils

# NOTE, does no halo updates, does not need to be a Parallel test, but doing so here to make the interface match fv_dynamics. could add support to the TranslateFortranData2Py class
class TranslateFVSubgridZ(ParallelTranslateBaseSlicing):
    inputs = {
        "delp": {
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa",
        },
        "delz": {
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m",
        },
        "pe": {
            "name": "interface_pressure",
            "dims": [fv3util.X_DIM, fv3util.Z_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "Pa",
            "n_halo": 1,
        },
        "pkz": {
            "name": "finite_volume_mean_pressure_raised_to_power_of_kappa",
            "units": "unknown",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "n_halo": 0,
        },
        "peln": {
            "name": "logarithm_of_interface_pressure",
            "units": "ln(Pa)",
            "dims": [fv3util.X_DIM, fv3util.Z_INTERFACE_DIM, fv3util.Y_DIM],
            "n_halo": 0,
        },
        "pt": {
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
        },
        "ua": {
            "name": "x_wind_on_a_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "va": {
            "name": "y_wind_on_a_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "w": {
            "name": "vertical_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "qvapor": {
            "name": "specific_humidity",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qliquid": {
            "name": "cloud_water_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qice": {
            "name": "ice_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qrain": {
            "name": "rain_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qsnow": {
            "name": "snow_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qgraupel": {
            "name": "graupel_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qo3mr": {
            "name": "ozone_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qsgs_tke": {
            "name": "turbulent_kinetic_energy",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m**2/s**2",
        },
        "qcld": {
            "name": "cloud_fraction",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
        },
        "u_dt": {
            "name": "x_wind_tendency",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s**2",
        },
        "v_dt": {
            "name": "y_wind_tendency",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s**2",
        },
        "te_1": {
            "name": "te_1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
        },
        "gzh_1": {
            "name": "gzh_1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
        },
        "gz_1": {
            "name": "gz_1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
        },
        "cvm_1": {
            "name": "cvm_1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
        },
        "u0_2": {
            "name": "u0_2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
        },
        "nq": {"dims": []},
        "dt": {"dims": []},
      
    }
    outputs = inputs.copy()
    '''
    outputs = { "ua": {
            "name": "x_wind_on_a_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
     "qliquid": {
            "name": "cloud_water_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },} #inputs.copy()
    '''

    for name in (
            "nq",
            "dt",
            "pe", "peln", "delp", "delz", "pkz",
            "u0_2", "cvm_1", "gz_1",  "gzh_1",  "te_1"
    ):
        outputs.pop(name)

    def __init__(self, grids, *args, **kwargs):
        super().__init__(grids, *args, **kwargs)
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "delp": {},
            "delz": {},
            "pkz": grid.compute_dict(),
            "ua": {},
            "va": {},
            "w": {},
            "pt": {},
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qo3mr": {},
            "qsgs_tke": {},
            "qcld": {},
            "u_dt": {},
            "v_dt": {},
            #"cpm_1": {'istart': 3, 'jstart':3, 'axis': 0, 'kstart': 3},
            "cvm_1": {'istart': 3, 'jstart':3, 'axis': 0, 'kstart': 3},
            "te_1": {'istart': 3, "dummy_axes": [1]},
            "gz_1": {'istart': 3, "dummy_axes": [1]},
            "gzh_1": {'istart': 3, 'jstart':3, 'axis': 0, 'kstart': 3},
            "u0_2": {'istart': 3, "dummy_axes": [1]},
          
        }

        self._base.out_vars = self._base.in_vars["data_vars"].copy()
        #self._base.out_vars = {"ua":{}, "qliquid":{}}
        for var in ["pe", "peln", "delp", "delz", "pkz",  "u0_2", "cvm_1", "gz_1",  "gzh_1",  "te_1"]:
            self._base.out_vars.pop(var)
        self.max_error = 1e-14

        self.ignore_near_zero_errors = {}
        for qvar in utils.tracer_variables:
            self.ignore_near_zero_errors[qvar] = True

    def compute_parallel(self, inputs, communicator):
        return self.compute_sequential(inputs)

    def compute_sequential(self, inputs):
        state = self.state_from_inputs(inputs)
        fv_subgridz.compute(state, inputs["nq"], inputs["dt"])
        outputs = self.outputs_from_state(state )
        return outputs

