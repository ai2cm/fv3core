# Contributing

FV3core is being actively developed by Vulcan Inc. Please contact us if there is interest
in making contributions. Credit will be given to contributors by adding their names
to the `CONTRIBUTORS.rst` file.


## Linting

The linter may change over time, but you should always be able to correct your code (if you have pip installed requirements_lint.txt locally) using `make lint`, or to correct all files, including those not yet in version control `make lint_all`

We manage the list of syntax requirements using [pre-commit](https://pre-commit.com/). This:
   - runs formatting and compliance checks for you
   - is required to pass to merge a PR

This will call [Black](https://github.com/ambv/black) which reformats your code to a set of accepted
standards.  Pre-commit now includes other adjustments and checks listed in the
.pre-commit-config.yaml file, including Flake8.

Exceptions to Flake8 standards:

Allow (ignore rule):

- W503 line break before binary operator
    - We should choose whether to ignore W503 or W504, and only ignore one
- E203 whitespace before ‘:'
    - Needs to be ignored to be consistent with black
- E302 Expected 2 blank lines, found 0
- F841 local variable is assigned to but never used
    - Probably have to ignore because that’s how stencils give outputs
    - Can avoid if all stencil assignments use square brackets on the left
        - out[0, 0, 0] or out[idx]/out[curr]/out[something]

Flake8 rules not listed will be enforced unless we find a need not to enforce them. Since documentation does go out of date, please consult the entry in `setup.cfg` for the most up-to-date requirements and update this document if you notice a difference.

## Style

The first version of the dycore was written with minimal metadata and typing, motivated
primarily by matching the regression data produced by the Fortran version of the code
using the numpy backend. We are now actively refactoring for performance with multiple
backends (cpu and gpu), moving code that still does computations in Python into GT4py
stencils and merging stencils together with the introduction of enabling features in
GT4py (such as regions). While we do that, clarifying the operation of the model and what the variables are
will both help make the model easier to read and reduce errors as we move around long
lists of argument variables.

Specifically, we want to start adding as makes sense:
- Type hints on python functions (see fv3core/utils/typing.py and below)
- More descriptive types on stencil definitions (fv3core/utils/typing.py)
- Docstrings on outward facing python functions: describe what methods are doing, describe
the intent (in, out, inout) of the function arguments

### Docstrings
These should aid us in refactoring and understanding what a function is doing. If it is not completely understood yet what is happening, what is known can be written along with a TODO to indicate it is incomplete.
e.g. vorticitytransport_cgrid:

"""Update the C-Grid zonal and meridional velocity fields.

    Args: uc: x-velocity on C-grid (inout)
          vc: y-velocity on C-grid (inout)
          vort_c: Vorticity on C-grid (inout)
	  ke_c: kinetic energy on C-grid (inout)
	  v: y-velocity on D-grid (inout)
	  u: x-velocity on D-grid (inout)
	  dt2: timestep (input)

"""


### Python functions
These should mostly be light workflow wrappers calling gt4py stencils, though currently
exceptions exist where python code does computations on data fields.

Original convention:
    def compute(var1, var2, var3, param1, param2, param3):

Order of arguments does not actually matter, but generally follows the convention of listing 3d fields first, followed by parameters, as is required by gt4py stencil functions.

New convention: make use of  fv3core/utils/typing.py to specify fields, also typehint any function
outputs.

For example:
    def compute(var1: FloatField, var2:IntField, var3: BoolField,
          param1: float_type, param2: int_type, param3: bool_type)

    """
    Describe what is being computed by this method

    Args:
      var1 (inout): description of data field
      var2 (in): description of int field
      ...

    """

Or another example using a gt4py_utils method:
Old convention:
def make_storage_from_shape(shape, origin, dtype, init=True):

New convention:

    def make_storage_from_shape(shape: Tuple[int, int, int], origin:
        Tuple[int, int, int] = origin, *, dtype: DTypes = np.float64, init: bool = True, mask:
        Tuple[bool, bool, bool] = (True, True, True), ) -> Field:

- see this method in gt4py_utils.py for its docstring as an example.
- We will prioritize adding typing to methods called by other modules, not every internal
  method needs this level of specification.
- Internal functions that are likely to be inlined into a larger stencil do not need this if it will just be removed shortly.

### Stencil functions

We currently have a custom decorator `@gtstencil` defined in
fv3core/decorators.py that helps set default external arguments such as "backend" and
rebuild" and provides the global namelist to the stencils. The type of each variable
going into a stencil requires a type and the first version of the model used a shorthand
'sd' (storage data) to indicate a gt4py field storage.

Example:

    @gtstencil()
    def pt_adjust(pkz:sd, dp1: sd, q_con: sd, pt: sd):
        with computation(PARALLEL), interval(...):
            pt = pt * (1.0 + dp1) * (1.0 - q_con) / pkz

When we have compile-time variations of the same stencil with different external
parameters, a stencil decorator can be defined interactively using the alternative
syntax. This however is quite a jarring change in convention and thus we try to avoid this
at the moment (does occur in fxadv), and may have another solution in the future.

e.g.:

    def undecorated_python_method(u, v):
      from __externals__ import vi
          with computation(PARALLEL), interval(...):
	      u = vi * v def compute(u, v):
called with:

    decorator =gtscript.stencil( backend=backend, rebuild=rebuild. externals={"vi": vi})
    stencil = decorator(undecorated_python_method) stencil(u, v, origin=origin, domain=domain)

In the new convention replace "sd" with FloatField (or whatever the type is).

### Externals
If a scalar parameter is in the scope of a module, it can be used inside of a
stencil (do not need an explicit import), otherwise use `from __externals__ import var`
inside the stencil definition

### Namelist
Initially the namelist was imported from the fv3core/_config. Now the namelist gets imported into the externals of a stencil using the decorator, and a stencil can use the namelist SimpleNamespace if it is imported with `from
__externals__ import namelist` inside the stencil.

### GTScript functions
These use the gtscript decorator and the arguments do not include type
specifications. They will continue to not have type hinting.

e.g.:

    @gtscript.function
    def get_bl(al, q):


### Assertions
We can now include assertions of compile time variables inside of gtscript
functions with the syntax `assert __INLINED(namelist.grid_type < 3)`.

### State
Some outer functions include a 'state' object that is a SimpleNamespace of variables and a
comm object that is the CubedSphereCommunicator object enabling halo updates.  The 'state'
include pointers to gt4py storages for all variables used in the method. For fields that
experience a halo update, the state includes pointers to Quantity objects named '<storage
variable name>_quantity', which is a lightweight wrapper around the storage. This enables
using gt4py storages in stencils and quantities for halo updates, using the same memory
space.  A future refactor will simplify this convention, likely through the use of the
decorator and/or a GDP from GT4py that may allow Quantities to be used in stencils.

As we refactor, we may opt to use this convention more (or a similar one to avoid calling functions while relying on getting the order of a long list of variables correct), but should be considered as part of a refactor on a case-by-case basis.


### New Styles

Propose new style ideas to the team (or subset) with examples and description of how data
flow would be altered if relevant. Once an idea is accepted, open a PR with the idea
applied to a sample if possible (if not, correct the whole model), and update this doc to
reflect the new convention we all should incorporate as we refactor. Share news of this
update when the PR is accepted and merged, including guidelines for using the new
convention. Implementers and reviewers of new code changes should consider whether the new style should be applied at the same time so we can introduce this change in a piecemeal fashion rather than disrupting every active task.
