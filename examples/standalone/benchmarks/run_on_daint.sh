SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPTPATH")")")"
echo $ROOT_DIR
cd $ROOT_DIR
cd ..

rm -rf vcm_1.0
cp -r /project/s1053/install/venv/vcm_1.0/ .
# git clone -b feature/standalone git@github.com:VulcanClimateModeling/fv3core.git
cd fv3core/
git submodule update --init --recursive
cd ..
vcm_1.0/bin/python -m pip install fv3core/external/fv3gfs-util/
vcm_1.0/bin/python -m pip install fv3core/

vcm_1.0/bin/python -m pip install gitpython

cp -r /project/s1053/fv3core_serialized_test_data/7.0.0/c12_6ranks_standard/ test_data
tar -xf test_data/dat_files.tar.gz -C test_data


# Adapt batch script:
git clone https://github.com/VulcanClimateModeling/buildenv/
cp buildenv/submit.daint.slurm .
sed s/\<NAME\>/standalone/g submit.daint.slurm -i
sed s/\<NTASKS\>/6/g submit.daint.slurm -i
sed s/\<NTASKSPERNODE\>/6/g submit.daint.slurm -i
sed s/\<CPUSPERTASK\>/2/g submit.daint.slurm -i
sed s/#SBATCH\ --output=\<OUTFILE\>//g submit.daint.slurm -i
sed s/\<G2G\>//g submit.daint.slurm -i
sed -i 's#<CMD>#export PYTHONPATH=/scratch/snx3000/tobwi/timing/serialbox2_master/gnu/python:$PYTHONPATH\nvcm_1.0/bin/python fv3core/examples/standalone/runfile/from_serialbox.py test_data/ fv3core/examples/standalone/config/c12_6ranks_standard.yml 2#g' submit.daint.slurm

sbatch -C gpu submit.daint.slurm


#args:
#- #ranks
#- experiment
#- size
#- dir to store
