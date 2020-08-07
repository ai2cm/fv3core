#!/bin/bash


# some global variables
maxsleep=9000

# check presence of env directory
envloc=`pwd`"/.jenkins"

# Download the env
. ${envloc}/env.sh

# setup module environment and default queue
if [ ! -f ${envloc}/env/machineEnvironment.sh ] ; then
    echo "Error 1201 test.sh ${LINENO}: Could not find ${envloc}/env/machineEnvironment.sh"
    exit 1
fi
. ${envloc}/env/machineEnvironment.sh

# load machine dependent functions
if [ ! -f ${envloc}/env/env.${host}.sh ] ; then
    exitError 1202 ${LINENO} "could not find ${envloc}/env/env.${host}.sh"
fi
. ${envloc}/env/env.${host}.sh

# load slurm tools
if [ ! -f ${envloc}/env/slurmTools.sh ] ; then
    exitError 1203 ${LINENO} "could not find ${envloc}/env/slurmTools.sh"
fi
. ${envloc}/env/slurmTools.sh

# check if SLURM script exists
script="${envloc}/env/submit.${host}.slurm"
test -f ${script} || exitError 1252 ${LINENO} "cannot find script ${script}"

# define command
cmd="module load sarus\n${mpilaunch} sarus run --mount=type=bind,source=/scratch/snx3000/rgeorge/test_data/c12_6ranks_standard,destination=/test_data load/library/fv3core:latest pytest --data_path=/test_data -v -s -rsx /fv3core/tests"

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
