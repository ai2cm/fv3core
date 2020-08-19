#!/bin/bash
envloc=$1
BACKEND=$2
EXPNAME=$3
ARGS="-v -s -rsx --backend=${BACKEND}"

maxsleep=9000
if [ "`hostname | grep daint`" != "" ] ; then
. ${envloc}/env/machineEnvironment.sh
# check if SLURM script exists
script="${envloc}/env/submit.${host}.slurm"
test -f ${script} || exitError 1252 ${LINENO} "cannot find script ${script}"


# load slurm tools
if [ ! -f ${envloc}/env/slurmTools.sh ] ; then
    exitError 1203 ${LINENO} "could not find ${envloc}/env/slurmTools.sh"
fi
. ${envloc}/env/slurmTools.sh

# load gcloud (TODO put this into buildenv)
module add /project/d107/install/modulefiles/
module load gcloud
# get the test data version from the Makefile
FORTRAN_VERSION=`grep "FORTRAN_SERIALIZED_DATA_VERSION=" Makefile  | cut -d '=' -f 2`
DATA_DIR="/scratch/snx3000/rgeorge/fv3core_fortran_data/${FORTRAN_VERSION}/${EXPNAME}/"
PROJECT_DATA_DIR="/project/d107/fv3core_serialized_test_data/${FORTRAN_VERSION}/${EXPNAME}/"
# sync the test data if it does not live in /scratch
if [ ! -d ${DATA_DIR} ] ; then
    TEST_DATA_HOST=${PROJECT_DATA_DIR} make sync_test_data
    mkdir -p ${DATA_DIR}
    cp -r ${PROJECT_DATA_DIR}/* ${DATA_DIR}
    TEST_DATA_HOST=${DATA_DIR} make unpack_test_data
fi
cd $SCRATCH
mkdir -p $EXPNAME
cd $EXPNAME
export FV3_CONTAINER=fv3core
export TAR_FILE=fv3core.tar
module load daint-gpu
module load sarus
gsutil copy gs://vcm-fv3core/${TAR_FILE} .
sarus load ./${TAR_FILE} ${FV3_CONTAINER}
# define command
cmd="${mpilaunch} sarus run --mount=type=bind,source=${DATA_DIR},destination=/test_data load/library/${FV3_CONTAINER}:latest pytest --data_path=/test_data ${ARGS} /fv3core/tests"

# setup SLURM job
out="fv3core_${BUILD_ID}.out"
/bin/sed -i 's|<NAME>|jenkins-fv3core-regression-sarus|g' ${script}
/bin/sed -i 's|<NTASKS>|1|g' ${script}
/bin/sed -i 's|<NTASKSPERNODE>|'"${nthreads}"'|g' ${script}
/bin/sed -i 's|<CPUSPERTASK>|1|g' ${script}
/bin/sed -i 's|<OUTFILE>|'"${out}"'|g' ${script}
/bin/sed -i 's|<CMD>|'"${cmd}"'|g' ${script}
/bin/sed -i 's|<PARTITION>|'"cscsci"'|g' ${script}

# submit SLURM job
launch_job ${script} ${maxsleep}
if [ $? -ne 0 ] ; then
  exitError 1251 ${LINENO} "problem launching SLURM job ${script}"
fi

# echo output of SLURM job
cat ${out}
rm ${out}
cd $SCRATCH
rm -r $EXPNAME
else
    export EXPERIMENT=${EXPNAME}
    make tests TEST_ARGS="${ARGS}"
    make tests_mpi TEST_ARGS="${ARGS}"
fi
