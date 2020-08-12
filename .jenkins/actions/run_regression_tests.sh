#!/bin/bash
envloc=$1
BACKEND=$2
EXPERIMENT=$3
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

DATA_DIR="/scratch/snx3000/rgeorge/test_data/c12_6ranks_standard"
# define command
cmd="module load sarus\n${mpilaunch} sarus run --mount=type=bind,source=${DATA_DIR},destination=/test_data load/library/fv3core:latest pytest --data_path=/test_data ${ARGS} /fv3core/tests"

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

else
    make tests TEST_ARGS="${ARGS}"
    make tests_mpi TEST_ARGS="${ARGS}"
fi
