# How to get performance numbers

## Daint
The run_on_daint script gets arguments:
- Timesteps
- Ranks
- backend
- (target directory)
- (data directory)
it will output a json file with times into the target dir. The first timestep is counted in init, the rest in main loop.
**Example:**
`examples/standalone/benchmarks/run_on_daint.sh 60 6 gtx86`
## Locally
Running examples/standalone/runfile/dynamics.py with the following arguments:
- data dir : where your serialized data is (already unpacked) with a file input.yml as well as input.nml as the namelist in there
- timestep: number of timesteps
- backend: chose the backend
**Example:**
`examples/standalone/runfile/dynamics.py  test_data/ 60 gtx86`
