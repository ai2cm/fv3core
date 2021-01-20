#!/bin/bash

## Arguments:
# $1: number of timesteps to run
# $2: number of ranks to execute with (ensure that this is compatible with fv3core)
# $3: path to the data directory that should be run
#############################################
# Example syntax:
# ./run_on_daint.sh 60 6

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
echo $ROOT_DIR

# check sanity of environment
test -n "$1" || exitError 1001 ${LINENO} "must pass a number of timesteps"
timesteps="$1"
test -n "$2" || exitError 1002 ${LINENO} "must pass a number of ranks"
ranks="$2"

target_dir="$3"
if [ -z "$3" ]
  then
    target_dir="$ROOT_DIR"
fi

data_path="$4"
if [ -z "$4" ]
  then
    data_path="/project/s1053/fv3core_serialized_test_data/7.0.0/c12_6ranks_standard/"
fi


# set up the virtual environment
cd $ROOT_DIR
cd ..
rm -rf vcm_1.0
cp -r /project/s1053/install/venv/vcm_1.0/ .
cd fv3core/
git submodule update --init --recursive
cd ..
vcm_1.0/bin/python -m pip install fv3core/external/fv3gfs-util/
vcm_1.0/bin/python -m pip install fv3core/
vcm_1.0/bin/python -m pip install gitpython

# set up the experiment data
cp -r $data_path test_data
tar -xf test_data/dat_files.tar.gz -C test_data


# Adapt batch script:
git clone https://github.com/VulcanClimateModeling/buildenv/
cp buildenv/submit.daint.slurm .
sed s/\<NAME\>/standalone/g submit.daint.slurm -i
sed s/\<NTASKS\>/$ranks/g submit.daint.slurm -i
sed s/\<NTASKSPERNODE\>/$ranks/g submit.daint.slurm -i
sed s/\<CPUSPERTASK\>/1/g submit.daint.slurm -i
sed s/#SBATCH\ --output=\<OUTFILE\>//g submit.daint.slurm -i
sed s/\<G2G\>//g submit.daint.slurm -i
sed -i "s#<CMD>#export PYTHONPATH=/project/c14/install/daint/serialbox2_master/gnu/python:\$PYTHONPATH\nsrun vcm_1.0/bin/python fv3core/examples/standalone/runfile/from_serialbox.py test_data/ fv3core/examples/standalone/config/c12_6ranks_standard.yml $timesteps#g" submit.daint.slurm
cat submit.daint.slurm

# execute on a gpu node
sbatch -C gpu submit.daint.slurm
wait
cd $ROOT_DIR/..
ls
cp *.json $target_dir
