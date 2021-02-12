
## Purpose

PR addresses the "TODOs" within the Saturation stencil, including removal of temporaries, incorporating function calls within conditionals when appropriate, using min/max functions when appropriate, and using function calls where previous versions of GT4Py would not allow.  Also, Field-type arguments for stencils are declared using "FloatField" from "fv3core.utils.typing" instead of using "utils.sd" from "fv3core.utils.gt4py_utils".

## Code changes:

Code changes are within /fv3core/stencils/saturation_adjustment.py

