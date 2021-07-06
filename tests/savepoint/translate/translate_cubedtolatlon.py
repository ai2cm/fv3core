import fv3core._config as spec
import fv3gfs.util as fv3util
from fv3core.stencils.c2l_ord import CubedToLatLon
from fv3core.testing import ParallelTranslate2Py


class TranslateCubedToLatLon(ParallelTranslate2Py):
    inputs = {
        "u": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
    }

    def __init__(self, grids):
        super().__init__(grids)
        grid = grids[0]
        self._base.in_vars["data_vars"] = {"u": {}, "v": {}, "ua": {}, "va": {}}
        self._base.out_vars = {
            "ua": {},
            "va": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
        }

    def compute_parallel(self, inputs, communicator):
        inputs["comm"] = communicator
        inputs = self.state_from_inputs(inputs)
        result = self._base.compute_from_storage(inputs)
        quantity_result = self.outputs_from_state(result)
        result.update(quantity_result)
        for name, data in result.items():
            if isinstance(data, fv3util.Quantity):
                result[name] = data.storage
        # Create stencil object
        self.cubedToLatLon = CubedToLatLon(
            communicator, spec.namelist, communicator, result["u"], result["v"]
        )
        # Run
        self.cubedToLatLon(**result)
        result.update(self._base.slice_output(result))
        return result
