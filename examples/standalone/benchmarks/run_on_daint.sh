#!/bin/bash

## Arguments:
# $1: number of timesteps to run
# $2: number of ranks to execute with (ensure that this is compatible with fv3core)
# $3: backend to use in gt4py
# $4: target directory to store the output in
# $5: path to the data directory that should be run
#############################################
# Example syntax:
# ./run_on_daint.sh 60 6 gtx86

exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPTPATH")")")"
echo "scriptpath is $SCRIPTPATH"
echo "root dir is : $ROOT_DIR"


# check sanity of environment
test -n "$1" || exitError 1001 ${LINENO} "must pass a number of timesteps"
timesteps="$1"
test -n "$2" || exitError 1002 ${LINENO} "must pass a number of ranks"
ranks="$2"

backend="$3"
if [ -z "$3" ]
  then
    backend="numpy"
fi

target_dir="$4"
if [ -z "$4" ]
  then
    target_dir="$ROOT_DIR"
fi

data_path="$5"
if [ -z "$5" ]
  then
    data_path="/project/s1053/fv3core_serialized_test_data/7.0.0/c12_6ranks_standard/"
fi


# set up the virtual environment
cd $ROOT_DIR
rm -rf vcm_1.0

echo "copying in the venv"
cp -r /project/s1053/install/venv/vcm_1.0/ .
git submodule update --init --recursive
echo "install requirements"
vcm_1.0/bin/python -m pip install external/fv3gfs-util/
vcm_1.0/bin/python -m pip install .
vcm_1.0/bin/python -m pip install gitpython

# set up the experiment data
cp -r $data_path test_data
tar -xf test_data/dat_files.tar.gz -C test_data

# set the environment
git clone https://github.com/VulcanClimateModeling/buildenv/
source buildenv/machineEnvironment.sh
source buildenv/env.${host}.sh

# Adapt batch script:
cp buildenv/submit.daint.slurm .
sed s/\<NAME\>/standalone/g submit.daint.slurm -i
sed s/\<NTASKS\>/$ranks/g submit.daint.slurm -i
sed s/\<NTASKSPERNODE\>/$ranks/g submit.daint.slurm -i
sed s/\<CPUSPERTASK\>/1/g submit.daint.slurm -i
sed s/#SBATCH\ --output=\<OUTFILE\>//g submit.daint.slurm -i
sed s/00:45:00/01:30:00/g submit.daint.slurm -i
sed s/\<G2G\>//g submit.daint.slurm -i
sed -i "s#<CMD>#export PYTHONPATH=/project/c14/install/daint/serialbox2_master/gnu/python:\$PYTHONPATH\nsrun vcm_1.0/bin/python examples/standalone/runfile/dynamics.py test_data/ examples/standalone/config/c12_6ranks_standard.yml $timesteps $backend#g" submit.daint.slurm
cat submit.daint.slurm

# execute on a gpu node
sbatch -W -C gpu submit.daint.slurm
wait
cp *.json $target_dir
