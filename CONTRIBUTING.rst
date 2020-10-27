============
Contributing
============

FV3core is being actively developed by Vulcan Inc. Please contact us if there is interest
in making contributions. Credit will be given to contributors by adding their names
to the ``AUTHORS.rst`` file.


Development guidelines
----------------------

Linting
~~~~~~~

The linter will become more strict over time, but you should always be able to correct (or
at minimum learn what is incorrect) your code using `make lint`.

We manage the list of syntax requirements using `pre-commit <https://pre-commit.com/>`__. This:
   - runs formatting and compliance checks for you
   - is required to pass to merge a PR
This initially was a call to `Black
<https://github.com/ambv/black>`__ which reformats your code to a set of accepted
standards.  Pre-commit now includes other adjustments and checks listed in the
.pre-commit-config.yaml file.

When we add Flake 8 to pre-commit, these are the rules that are most important to us to
consider:

Allow (ignore rule):

- W503 line break before binary operator
    - We should choose whether to ignore W503 or W504, and only ignore one
- E302 whitespace before ‘:'
    - Needs to be ignored to be consistent with black

Add in a future PR but not immediately:

- F841 local variable is assigned to but never used
    - Probably have to ignore because that’s how stencils give outputs
    - Can avoid if all stencil assignments use square brackets on the left
        - out[0, 0, 0] or out[idx]/out[curr]/out[something]

Keep the rule:

- W504 line break after binary operator
    - ignoring W503 instead
- F401 imported but unused
    - Good for understanding what a file’s dependencies are
- F821 undefined name
    - To fix requires importing names explicitly
    - Useful because it catches real bugs
- E501 line too long (> 88 characters)
    - Should be included, these are lines black can’t handle automatically
- E265 block comment should start with “# “ and E262 inline comment should start with “# “


File Structure / Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 - The main functionality of the FV3 dynamical core, which has been ported from the
   Fortran version in the fv3gfs-fortran repo, is defined using GT4py stencils and python
   'compute' functions in fv3core/stencils.
 - The core is comprised of units of calculations defined for regression testing. These
   were initially generally separated into distinct files in
   fv3core/stencils with corresponding files in tests/translate/translate_<unit>.py
   defining the translation of variables from Fortran to Python. Exceptions exist in cases
   where topical and logical grouping allowed for code reuse. As refactors optimize the
   model, these units may be merged to occupy the same files and even methods/stencils, but
   the units should still be tested separately.
 - The core has most of its calculations happening in GT4py stencils, but there are still
   several instances of operations happening in Python directly, which will need to be
   replaced with GT4py code for optimal performance.
 - The 'units' fv_dynamics and fv_subgridz can be called by fv3gfs-wrapper to run the
   FV3core model using Pythong wrapped fortran for code not ported to GT4py at the moment.
 - The namelist and grid are global variables defined in fv3core/_config.py
   - The namelist is 'flattened' so that the grouping name of the option is not required
     to access the data (we may want to change this)
   - The grid variables are mostly 2d variables that have been replicated in the z
     dimension as gt4py storages for use in stencils
   - The grid object also contains domain and layout information relevant to the current
     rank being operated on.
 - Utility functions at fv3core/utils include:
      - gt4py_utils:
	 - default gt4py and model settings
	 - methods for generating gt4py storages
	 - methods for using numpy and cupy arrays in python functions that have not been
           put into GT4py
	 - methods for handling complex patterns that did not immediately map to gt4py,
           and will mostly be removed with future refactors (e.g. k_split_run)
	 - some general model math computations (e.g. great_circle_dist), that will
           eventually be put into gt4py with a future refactor
      - grid:
	 - A Grid class definition that provides information about the grid layout,
           current tile informationm access to grid variables used globally, and
           convenience methods related to tile indexing, origins and domains commonly used
         - A grid is defined for each MPI rank (minimum 6 ranks, 1 for each tile face of
           the cubed sphere grid represnting the whole Earth)
	 - Also provides functionality for generating a Quantity object (used to interface
           with the fv3gfs-wrapper, that allows us to run the full model, not just the
           dynamical core)
      - corners: port of corner calculations, initially direct Python calculations, being
        replaced with GT4py gtscript functions as the GT4py regions feature is implemented
      - mpi: a wrapper for importing mpi4py when available
      - global_constants.py: constants for use throughout the model
      - typing.py: Clean names for common types we use in the model. This is new and
        hasn't been adopted throughout the model yet, but will eventually be our
        standard. A shorthand 'sd' has been used in the intial version.
 - `tests/` currently includes a framework for translating fields serialized (using
   Serialbox from GridTools) from a Fortran run into gt4py storages that can be inputs to
   fv3core unit computations, and compares the results of the ported code to serialized
   data following a unit computation.
 - `docker/` provides Dockerfiles for building a repeatable environment in which to run the
   core
 - `external/`: a directory for submoduled repos that provide essential functionality
 - The build system uses Makefiles following the convention of other repos within
   VulcanClimateModeling

Model Interface
~~~~~~~~~~~~~~~

 - Top level functions fv_dynamics and fv_sugridz can currenty only be run in parallel
   using mpi with a minimum of 6 ranks (there are a few other units that also require
   this, e.g. whenever there is a halo update involved in a unit)
   - These are the interface to the rest of the model and currently have different
     conventions than the rest of the model
   - A 'state' object (currently a SimpleNamespace) stores pointers to the allocated data
     fields
 - Most functions within dyn_core can be run sequentially per rank
 - Currently a list of ArgSpecs must decorate an interface function, where each ArgSpec
   provides useful information about the argument, e.g.: @state_inputs( ArgSpec("qvapor",
   "specific_humidity", "kg/kg", intent="inout")
    - The format is (fortran_name, long_name, units, intent)
    - We currently provide a duplicate of most of the metadata in the specification of the
      unit test, but that may be removed eventually.
 - Then the function itself, e.g. fv_dynamics, has arguments of 'state', 'comm' (the
   communicator) and all of the scalar parameters being provided.



Style
~~~~~


The first version of the dycore was written with minimal metadata and typing, motivated
primarily by matching the regression data produced by the Fortran version of the code
using the numpy backend. We are now actively refactoring for performance with multiple
backends (cpu and gpu), moving code that still does computations in Python into GT4py
stencils and merging stencils together with the introduction of enabling features in
GT4py. While we do that, clarifying the operation of the model and what the variables are
will both help make the model easier to read and reduce errors as we move around long
lists of argument variables.

Specifically, we want to start adding as makes sense:
- Type hints on python functions (see typing.py and below)
- More descriptive types on stencil definitions (typing.py)
- Docstrings on outward facing python functions: describe what methods are doing, describe
the intent (in, out, inout) of the function arguments
e.g. vorticitytransport_cgrid:
"""Update the C-Grid zonal and meridional velocity fields.

    Args: uc: x-velocity on C-grid (inout) vc: y-velocity on C-grid (inout) vort_c:
        Vorticity on C-grid (inout) ke_c: kinetic energy on C-grid (inout) v: y-velocit on
        D-grid (inout) u: x-velocity on D-grid (inout) dt2: timestep (input) """


Python functions (should mostly be light wrappers calling gt4py stencils, though currently
exceptions exist where python code does computations on data fields):
Original convention:
def compute(var1, var2, var3, param1, param2, param3):
Order of arguments did not matter
too much, but generally follows the convention of listing 3d fields first, followed by
parameters.

New convention: make use of typing.py to specify fields, also typehint any function
outputs. For example:
def compute(var1: FloatField, var2:IntField, var3: BoolField,
param1: float_type, param2: int_type, param3: bool_type):
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
def make_storage_from_shape( shape: Tuple[int, int, int], origin:
Tuple[int, int, int] = origin, *, dtype: DTypes = np.float64, init: bool = True, mask:
Tuple[bool, bool, bool] = (True, True, True), ) -> Field:
- see this method in gt4py_utils.py for its docstring as an example.
- We will prioritize adding typing to methods called by other modules, not every internal
  method needs this level of specification.

Stencil functions: We currently have a custom decorator @gtstencil defined in
fv3core/decorators.py that helps set default external arguments such as "backend" and
rebuild" and provides the global namelist to the stencils. The type of eaach variable
going into a stencil requires a type and the first version of the model used a shorthand
'sd' (storage data) to indicate a gt4py field storage.
@gtstencil()
def pt_adjust(pkz:sd, dp1: sd, q_con: sd, pt: sd):
    with computation(PARALLEL), interval(...): pt = pt *
        (1.0 + dp1) * (1.0 - q_con) / pkz

When we have compile-time variations of the same stencil with different external
parameters, a stencil decorator can be defined interactively using the alternative
syntax. This however is quite a jarring change in convention and thus we try to avoid this
at the moment (does occur in fxadv), and may have another solution in the future.
def undecorated_python_method(u, v):
    from __externals__ import vi
        with computation(PARALLEL), interval(...):
	    u = vi * v def compute(u, v):
decorator =gtscript.stencil( backend=backend, rebuild=rebuild. externals={"vi": vi})
stencil = decorator(undecorated_python_method) stencil(u, v, origin=origin, domain=domain)

In the new convention replace "sd" with FloatField (or whatever the type is).

Externals:
If a scalar parameter is in the scope of a module, it can be used inside of a
stencil (do not need an explicit import), otherwise use "from __externals__ import var"
inside the stencil definition

Namelist: Initially the namelist was imported from the
fv3core/_config. Now the namelist gets imported into the externals of a stencil using the
decorator, and a stencil can use the namelist SimpleNamespace if it is imported with from
__externals__ import namelist inside the stencil

GTScript functions:
These use the gtscript decorator and the arguments do not include type
specifications. They will continue to not have type hinting.
@gtscript.function
def get_bl(al, q):


Assertions
We can now include assertions of compile time variables inside of gtscript
functions with the syntax: assert __INLINED(namelist.grid_type < 3)

State
Some functions include a 'state' object that is a SimpleNamespace of variables and a
comm object that is the CubedSphereCommunicator object enabling halo updates.  The 'state'
include pointers to gt4py storages for all variables used in the method. For fields that
experience a halo update, the state includes pointers to Quantity objects named '<storage
variable name>_quantity', which is a lightweight wrapper around the storage. This enables
using gt4py storages in stencils and quantities for halo updates, using the same memory
space.  A future refactor will simplify this convention, likely through the use of the
decorator and/or GDP-3 from GT4py that may allow Quantities to be used in stencils..


New Styles
~~~~~

Propose new style ideas to the team (or subset) with examples and description of how data
flow would be altered if relevant. Once an idea is accepted, open a PR with the idea
applied to a sample if possible (if not, correct the whole model), and update this doc to
reflect the new convention we all should incorporate as we refactor. Share news of this
update when the PR is accepted and merged, including guidelines for utsing the new
convention.

Porting Conventions
~~~~~

Generation of regression data occurs in the fv3gfs-fortran repo
(https://github.com/VulcanClimateModeling/fv3gfs-fortran) with serialization statements
and a build procedure defined in tests/serialized_test_data_generation. The version of
data this repo currently tests against is defined in FORTRAN_SERIALIZED_DATA_VERSION in
the Makefile. Fields serialized are defined in Fortran code with serialization comment
statements such as: !$ser savepoint C_SW-In !$ser data delpcd=delpc delpd=delp ptcd=ptc
Where the name being assigned is the name the fv3core uses to identify the variable in the
test code. When this name is not equal to the name of the variable, this was usually done
to avoid conflicts with other parts of the code where the same name is used to reference a
differently sized field.

The majority of the logic for translating from data serialized from Fortran to something
that can be used by Python, and the comparison of the results, is encompassed by the main
Translate class in the tests/translate/translate.py file. Any units not involving a halo
update can be run using this framework, while those that need to be run in parallel can
look to the ParallelTranslate class as the parent class in
tests/translate/parallel_translate.py. These parent classes provide generally useful
operations for translating serialized data between Fortran and Python specifications, and
for applying regression tests.  A new unit test can be defined as a new child class of one
of these, with a naming convention of Translate<Savepoint Name> where "Savepoint Name" is
the name used in the serialization statements in the Fortran code, without the "-In" and
"-Out" part of the name. A translate class can usually be minimally specify the input and
output fields. Then, in cases where the parent compute function is insuffient to handle
the complexity of either the data translation or the compute function, the appropriate
methods can be overridden.

For Translate objects
  - The init function establishes the assumed translation setup for the class, which can
    be dynamically overridden as needed.
  - the parent compute function does:
    1. makes gt4py storages of the max shape (grid.npx+1, grid.npy+1, grid.npz+1) aligning
       the data based on the start indices specified. (gt4py requires data fields have the
       same shape, so in this model we have buffer points so all calculations can be done
       easily without worrying about shape matching)
    2. runs the compute function (defined in self.compute_func) on the input data storages
    3. slices the computed Python fields to be compared to fortran regression data
  - The unit test then uses a modified relative error metric to determine whether the unit
    passes
  - The init method for a Translate class:
    - the input ( self.in_vars["data_vars"]) and output(self.out_vars) variables are
      specified in dictionaries, where the keys are the name of the variable used in the
      model and the values are dictionaries specifying metadata for translation of
      serialized data to gt4py storages. The metadata that can be specied to override
      defaults are:
      - indices to line up data arrays into gt4py storages (which all get created as tha
        max possible size needed by all operations, for simplicity):
	 - "istart", "iend", "jstart", "jend", "kstart", "kend"
	 - These should be set using the 'grid' object available to the Translate object,
	   using equivalent index names as in the declaration of variables in the Fortran
	   code, e.g.  real:: cx(bd%is:bd%ie+1,bd%jsd:bd%jed ) should include
	   self.in_vars["data_vars"]["cx"] = {"istart": self.is_, "iend": self.ie + 1,
	   "jstart": self.jsd, "jend": self.jed,} There is only a limited set of Fortran
	   shapes declared, so abstractions defined in the grid can also be used, e.g.
	   self.out_vars["cx"] = self.grid.x3d_compute_domain_y_dict()

	  - Note that the variables, e.g. grid.is_ and grid.ie specify the 'compute'
    domain in the x direction of the current tile, equivalent to bd%is and bd%ie in the
    Fortran model EXCEPT that the Python variables are local to the current MPI rank (a
    subset of the tile face), while the Fortran values are global to the tile face. This
    is because these indices are used to slice into fields, which in Python is 0-based,
    and in Fortran is based on however the variables are declared. But, for the purposes
    of aligning data for computations and comparisons, we can match them in this
    framework.  shapes need to be defined in a dictionary per variable including "istart",
    "iend", "jstart", "jend", "kstart", "kend" that represent the shape of that variable
    as defined in the Fortran code. The default shape assumed if a variable is specified
    with an empty dictionary is isd:ied, jsd:jed, 0:npz - 1 inclusive, and variables that
    aren't that shape in the Fortran code need to have the 'start' indices specified for
    the in_vars dictionary , and 'start' and 'end' for the out_vars.
     - "serialname" can be used to specify a name used in the Fortran code declaration if
       we'd like the model to use a different name
     - 'kaxis': which dimension is the vertical direction. For most variables this is '2'
       and does not need to be specified. For Fortran variables that assign the vertical
       dimension to a different axis, this can be set to ensure we end up with 3d storages
       that have the vertical dimension where it is expected by GT4py.
     - 'dummy_axes': If set this will set of the storage to have singleton dimensions in
       the axes defined. This is to enable testing stencils where the full 3d data has not
       been collected and we want to run stencil tests on the data for a particular slice.
     - 'names_4d': If a 4d variable is being serialized, this can be set to specify the
       names of each 3d field. By default this is the list of tracers.

    - input variables that are scalars should be added to self.in_vars["parameters"]
    - self.compute_func is the name of the model function that should be run by the
      compute method in the translate class
    - self.max_error overrides the parent classes relative error threshold. This should
      only be changed when the reasons for non-bit reproducibility are understood.
    - self.max_shape sets the size of the gt4py storage created for testing
    - self.ignore_near_zero_errors[<varname>] = True: This is an option to let some fields
      pass with higher relative error if the absolute error is very small


For ParallelTranslate objects:
  - inputs and outputs are defined at the class level, and these include metadata such as
    the "name" (e.g. understandable name for the symbol), dimensions, units and
    n_halo(numb er of halo lines)
  - Both 'compute_sequential' and 'compute_parallel' method may be defined, where a mock
    communicator is used in the compute_sequential case
  - The parent assumes a state object for tracking fields and methods exist for
    translating from inputs to a state object and extracting the output variables from the
    state. It is assumed that Quantity objects are needed in the model method in order to
    do halo updates.
  - ParallelTranslate2Py is a slight variation of this used for many of the parallel units
    that do not yet utilize a state object and relies on the specification of the same
    index metadata of the Translate classes
  - ParallelTranslateBaseSlicing makes use of the state but relies on the Translate object
    of self._base, a Translate class object, to align the data before computing and
    comparing.
