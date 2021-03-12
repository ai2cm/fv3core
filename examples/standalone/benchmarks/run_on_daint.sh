#!/bin/bash

#############################################
# Example syntax:
# ./run_on_daint.sh 60 6 gtx86

## Arguments:
# $1: number of timesteps to run
# $2: number of ranks to execute with (ensure that this is compatible with fv3core)
# $3: backend to use in gt4py
# $4: path to the data directory that should be run
# $5: (optional) arguments to pass to python invocation
# $6: (optional) arguments to pass to dynamics.py invocation

set -e
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
NTHREADS=12

# check sanity of environment
test -n "$1" || exitError 1001 ${LINENO} "must pass a number of timesteps"
timesteps="$1"
test -n "$2" || exitError 1002 ${LINENO} "must pass a number of ranks"
ranks="$2"
test -n "$3" || exitError 1003 ${LINENO} "must pass a backend"
backend="$3"
test -n "$4" || exitError 1004 ${LINENO} "must pass a data path"
data_path="$4"
py_args="$5"
run_args="$6"

# get dependencies
cd $ROOT_DIR
git submodule update --init external/fv3gfs-util external/daint_venv

# set GT4PY version
cd $ROOT_DIR
export GT4PY_VERSION=`grep "GT4PY_VERSION=" docker/Makefile.image_names  | cut -d '=' -f 2`

# set up the virtual environment
echo "creating the venv"
if [ -d ./venv ] ; then rm -rf venv ; fi
cd $ROOT_DIR/external/daint_venv/
if [ -d ./gt4py ] ; then rm -rf gt4py ; fi
./install.sh $ROOT_DIR/venv
cd $ROOT_DIR
source ./venv/bin/activate

# install the local packages
echo "install requirements..."
pip install ./external/fv3gfs-util/
pip install .
pip list

# set the environment
if [ -d ./buildenv ] ; then rm -rf buildenv ; fi
git clone https://github.com/VulcanClimateModeling/buildenv/
cp ./buildenv/submit.daint.slurm compile.daint.slurm
cp ./buildenv/submit.daint.slurm run.daint.slurm

if git rev-parse --git-dir > /dev/null 2>&1 ; then
  githash=`git rev-parse HEAD`
else
  githash="notarepo"
fi

echo "Configuration overview:"
echo "    Root dir:         $ROOT_DIR"
echo "    Timesteps:        $timesteps"
echo "    Ranks:            $ranks"
echo "    Backend:          $backend"
echo "    Input data dir:   $data_path"
echo "    Threads per rank: $NTHREADS"
echo "    GIT hash:         $githash"
echo "    Python arguments: $py_args"
echo "    Run arguments:    $run_args"

echo "copying premade GT4Py caches"
split_path=(${data_path//\// })
experiment=${split_path[-1]}
sample_cache=.gt_cache_000000
if [ ! -d $(pwd)/${sample_cache} ] ; then
    premade_caches=/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$experiment/$backend
    if [ -d ${premade_caches}/${sample_cache} ] ; then
	cp -r ${premade_caches}/.gt_cache_0000* .
	find . -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/olifu\/jenkins_submit\/workspace\/fv3core-cache-setup\/backend\/$backend\/experiment\/$experiment\/slave\/daint_submit|$(pwd)|g" {} +
    fi
fi

echo "submitting script to do compilation"
# Adapt batch script to compile the code:
sed -i s/\<NAME\>/standalone/g compile.daint.slurm
sed -i s/\<NTASKS\>/$ranks/g compile.daint.slurm
sed -i s/\<NTASKSPERNODE\>/1/g compile.daint.slurm
sed -i s/\<CPUSPERTASK\>/$NTHREADS/g compile.daint.slurm
sed -i s/--output=\<OUTFILE\>/--hint=nomultithread/g compile.daint.slurm
sed -i s/00:45:00/03:30:00/g compile.daint.slurm
sed -i s/cscsci/normal/g compile.daint.slurm
sed -i s/\<G2G\>/export\ CRAY_CUDA_MPS=1/g compile.daint.slurm
sed -i "s#<CMD>#export PYTHONPATH=/project/s1053/install/serialbox2_master/gnu/python:\$PYTHONPATH\nsrun python examples/standalone/runfile/dynamics.py $data_path 1 $backend $githash --disable_halo_exchange#g" compile.daint.slurm

# execute on a gpu node
sbatch -W -C gpu compile.daint.slurm
wait
echo "compilation step finished"

echo "submitting script to do performance run"
# Adapt batch script to run the code:
sed -i s/\<NAME\>/standalone/g run.daint.slurm
sed -i s/\<NTASKS\>/$ranks/g run.daint.slurm
sed -i s/\<NTASKSPERNODE\>/1/g run.daint.slurm
sed -i s/\<CPUSPERTASK\>/$NTHREADS/g run.daint.slurm
sed -i s/--output=\<OUTFILE\>/--hint=nomultithread/g run.daint.slurm
sed -i s/00:45:00/00:40:00/g run.daint.slurm
sed -i s/cscsci/normal/g run.daint.slurm
sed -i s/\<G2G\>//g run.daint.slurm
sed -i "s#<CMD>#export PYTHONPATH=/project/s1053/install/serialbox2_master/gnu/python:\$PYTHONPATH\nsrun python $py_args examples/standalone/runfile/dynamics.py $data_path $timesteps $backend $githash $run_args#g" run.daint.slurm

# execute on a gpu node
sbatch -W -C gpu run.daint.slurm
wait

echo "performance run sucessful"
