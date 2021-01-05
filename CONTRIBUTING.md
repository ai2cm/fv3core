# Contributing

FV3core is actively developed by Vulcan, so please contact us if there is interest in making contributions in the near-term.
Contributors names will be added to [AUTHORS.rst](https://github.com/VulcanClimateModeling/fv3core/blob/master/AUTHORS.rst).

## Linting

Dependencies for linting are maintained in `requirements_lint.txt`, and can be installed and run with:

```shell
$ pip install -r requirements_lint.txt
$ make lint # This runs all the checkers
```

We manage the list of syntax requirements using [pre-commit](https://pre-commit.com/).
**This runs all checks and is required to pass as one of the continuous integration tests.**

The list of checkes includes `black`, `isort`, and `flake8`, among a few others.
Black is configured in `pyproject.toml`, and the others use `setup.cfg`.

We mostly use standard Flake8, but ignore the following rules:

- `W503: line break before binary operator`
    - We should choose whether to ignore W503 or W504, and only ignore one
- `E203: whitespace before ':'`
    - Needs to be ignored to be consistent with black
- `E302: Expected 2 blank lines, found 0`
- `F841: local variable is assigned to but never used`
    - Must ignore because gt4py stencils return outputs in place (no return statement)
    - Can avoid if all stencil assignments use square brackets on the left
        - `out[0, 0, 0]` or `out[idx]/out[curr]/out[something]`

Flake8 rules not listed will be enforced unless we find a need not to enforce them.
Since documentation does go out of date, please consult the entry in `setup.cfg` for the most up-to-date requirements and update this document if you notice a difference.

## Style

The first version of the dycore was written with minimal metadata and typing, motivated primarily by matching the regression data produced by the Fortran version of the code using the numpy backend.
We are now actively refactoring, moving code that still does computations in Python into GT4py stencils and merging stencils together with the introduction of enabling features in GT4py (such as regions).
While we do that, clarifying the operation of the model and what the variables are will both help make the model easier to read and reduce errors as we move around long lists of argument variables.

Specifically, we want to start adding the following where appropriate:
- Type hints on Python functions (see [`utils/typing.py`](https://github.com/VulcanClimateModeling/fv3core/blob/master/utils/typing.py) and below)
- More descriptive types on stencil definitions
- Docstrings on outward facing Python functions: describe what methods are doing, describe the intent (*in*, *out*, or *inout*) of the function arguments

### Docstrings
These should aid us in refactoring and understanding what a function is doing. If it is not completely understood yet what is happening, what is known can be written along with a `TODOC` to indicate it is incomplete.
For example:

```python
def stencil(...):
    """This is a short description that fits on one line.

    Here is a longer explanation of the assumptions and why this exists.

    TODOC: I do not fully understand why ke_c needs to be updated here.

    Args:
        uc: x-velocity on C-grid (inout)
        vc: y-velocity on C-grid (inout)
        vort_c: Vorticity on C-grid (inout)
        ke_c: kinetic energy on C-grid (inout)
        v: y-velocity on D-grid (inout)
        u: x-velocity on D-grid (inout)
        dt2: timestep (in)

    """
```


### Python functions
These should mostly be lightweight workflow wrappers calling gt4py stencils, though currently exceptions exist where Python code does computations on data fields.

Original convention is:
```python
def compute(var1, var2, var3, param1, param2, param3):
```

Order of arguments does not actually matter, but generally follows the convention of listing 3d fields first, followed by parameters, as is required by gt4py stencil functions.

New convention: make use of `fv3core/utils/typing.py` to specify fields, also type-hint any function
outputs.

For example:
```python
def compute(var1: FloatField, var2:IntField, var3: BoolField,
            param1: float_type, param2: int_type, param3: bool_type):
```

Another example
```python
def make_storage_from_shape(shape, origin, dtype, init=True):
```

Turns into
```python
    def make_storage_from_shape(
        shape: Tuple[int, int, int],
        origin: Tuple[int, int, int],
        *,
        dtype: DTypes = np.float64,
        init: bool = True,
    ) -> Field:
```

- We will prioritize adding typing to methods that are used by other modules.
  Not every internal method needs this level of specification.
- Internal functions that are likely to be inlined into a larger stencil do not need this if it will just be removed in the near-term.

### GT4Py stencils
FV3core defines a custom decorator `fv3core.gtstencil` defined in `decorators.py` that it uses to define stencils.
This eventually calls `gt4py.gtscript.stencil`, but sets default external arguments such as `backend`, and `rebuild` and provides the global namelist to the stencils as `namelist`.
The type of each input of a stencil requires a type and the first version of the model used a shorthand 'sd' (storage data) to indicate a 3D gt4py storage, such as

```python
@gtstencil()
def pt_adjust(pkz:sd, dp1: sd, q_con: sd, pt: sd):
    with computation(PARALLEL), interval(...):
```

Note that `fv3core.gtstencil` can be manually called on an undecorated stencil, but this is currently in general discouraged except when used internally.

In the refactoring of the dycore, we are using lower dimensional storages and different item types, so `sd` is insufficient to type these.
[`utils/typing.py`](https://github.com/VulcanClimateModeling/fv3core/blob/master/fv3core/utils/typing.py) defines various field types.
For example, `FloatField[IJ]` for a 2D field of default floating point values.

### Namelist
The `fv3core.gtstencil` decorator automatically makes `namelist` available, if `from __externals__ import namelist` is added at the top of the stencil or any stencil function.

### Assertions
We can now include assertions of compile time variables inside of gtscript functions with the syntax: `assert __INLINED(namelist.grid_type < 3)`.

### State
Some outer functions include a 'state' object that is a SimpleNamespace of variables and a `comm` object that is the `CubedSphereCommunicator` object enabling halo updates.
The `state` include pointers to gt4py storages for all variables used in the method.
For fields that experience a halo update, the state includes pointers to Quantity objects named `<storage variable name>_quantity`, which is a lightweight wrapper around the storage.
This enables using gt4py storages in stencils and quantities for halo updates, using the same memory space.
A future refactor will simplify this convention, likely through the use of the decorator and/or a GDP from GT4py that may allow Quantities to be used in stencils.

As we refactor, we may opt to use this convention more (or a similar one to avoid calling functions while relying on getting the order of a long list of variables correct), but should be considered as part of a refactor on a case-by-case basis.


### New styles
Propose new style ideas to the team (or subset) with examples and description of how data flow would be altered if relevant. Once an idea is accepted, open a PR with the idea applied to a sample if possible (if not, correct the whole model), and update this doc to reflect the new convention we all should incorporate as we refactor.
Share news of this update when the PR is accepted and merged, including guidelines for using the new convention.
Implementers and reviewers of new code changes should consider whether the new style should be applied at the same time so we can introduce this change in a piecemeal fashion rather than disrupting every active task.
