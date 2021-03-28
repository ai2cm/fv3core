import fv3core.stencils.fxadv as fxadv
import fv3core.utils.gt4py_utils as utils
from fv3core.testing import TranslateFortranData2Py


class TranslateFxAdv(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        utinfo = grid.x3d_domain_dict()
        vtinfo = grid.y3d_domain_dict()
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "ut": utinfo,
            "vt": vtinfo,
            "xfx_adv": grid.x3d_compute_domain_y_dict(),
            "crx_adv": grid.x3d_compute_domain_y_dict(),
            "yfx_adv": grid.y3d_compute_domain_x_dict(),
            "cry_adv": grid.y3d_compute_domain_x_dict(),
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {
            "ra_x": {"istart": grid.is_, "iend": grid.ie},
            "ra_y": {"jstart": grid.js, "jend": grid.je},
            "ut": utinfo,
            "vt": vtinfo,
        }
        for var in ["xfx_adv", "crx_adv", "yfx_adv", "cry_adv"]:
            self.out_vars[var] = self.in_vars["data_vars"][var]

    def compute_from_storage(self, inputs):
        grid = self.grid
        fxadv.fxadv_stencil(
            cosa_u=grid.cosa_u,
            cosa_v=grid.cosa_v,
            rsin_u=grid.rsin_u,
            rsin_v=grid.rsin_v,
            sin_sg1=grid.sin_sg1,
            sin_sg2=grid.sin_sg2,
            sin_sg3=grid.sin_sg3,
            sin_sg4=grid.sin_sg4,
            rdxa=grid.rdxa,
            rdya=grid.rdya,
            dy=grid.dy,
            dx=grid.dx,
            **inputs,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(),
        )

        fxadv.fxadv_x_edges(
            cosa_u=grid.cosa_u,
            cosa_v=grid.cosa_v,
            rsin_u=grid.rsin_u,
            rsin_v=grid.rsin_v,
            sin_sg1=grid.sin_sg1,
            sin_sg2=grid.sin_sg2,
            sin_sg3=grid.sin_sg3,
            sin_sg4=grid.sin_sg4,
            rdxa=grid.rdxa,
            rdya=grid.rdya,
            dy=grid.dy,
            dx=grid.dx,
            **inputs,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(),
        )

        fxadv.ut_corners(grid.cosa_u, grid.cosa_v,inputs["uc"], inputs["vc"], inputs["ut"], inputs["vt"], origin=(1, 1, 0),domain=grid.domain_shape_full(add=(-1, -1, 0)) )  
        fxadv.vt_corners(grid.cosa_u, grid.cosa_v,inputs["uc"], inputs["vc"], inputs["ut"], inputs["vt"], origin=(1, 1, 0),domain=grid.domain_shape_full(add=(-1, -1, 0)))
        fxadv.fxadv_stencil_prod(
            cosa_u=grid.cosa_u,
            cosa_v=grid.cosa_v,
            rsin_u=grid.rsin_u,
            rsin_v=grid.rsin_v,
            sin_sg1=grid.sin_sg1,
            sin_sg2=grid.sin_sg2,
            sin_sg3=grid.sin_sg3,
            sin_sg4=grid.sin_sg4,
            rdxa=grid.rdxa,
            rdya=grid.rdya,
            dy=grid.dy,
            dx=grid.dx,
            **inputs,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(),
        )
        inputs["ra_x"] = utils.make_storage_from_shape(
            inputs["uc"].shape, grid.compute_origin()
        )
        inputs["ra_y"] = utils.make_storage_from_shape(
            inputs["vc"].shape, grid.compute_origin()
        )
        fxadv.flux_divergence_area(
            area=grid.area,
            xfx_adv=inputs["xfx_adv"],
            yfx_adv=inputs["yfx_adv"],
            ra_x=inputs["ra_x"],
            ra_y=inputs["ra_y"],
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(),
        )
        return inputs
