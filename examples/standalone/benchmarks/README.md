# How to get performance numbers

The script `run_on_daint.sh` allows you to run a benchmark of the dynamical core standalone
`dynamics.py` on bare metal Piz Daint.

## Arguments

-   timesteps: Number of timesteps to execute (this includes the first one as a warm up step)
-   ranks: Number of ranks to run with
-   backend: choice of gt4py backend
-   data\_path: path to the directory containing the serialized test data
-   py\_args: (optional) arguments to pass to python invocation
-   run\_args: (optional) arguments to pass to the dynamics.py invocation

## Constraints

The data directory is expected to contain unpacked serialized data (serialized by serialbox).
The serialized data is also expected to have both the `input.nml` as well as the `*.yml` namelists present.

## Output

A `timing.json` file containing statistics over the ranks for execution time.
The first timestep is counted towards `init`, the rest of the timesteps are in `main loop`.
Total is inclusive of the other categories

## Example

`./examples/standalone/benchmarks/run_on_daint.sh 60 6 gtx86 <data_path>`
