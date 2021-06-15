make update_submodules_base
rm -rf test_venv
python -m venv test_venv
source ./test_venv/bin/activate
pip install --upgrade pip
pip install --upgrade wheel

# installation of fv3 dependencies
pip install cftime f90nml pandas pyparsing python-dateutil pytz pyyaml xarray zarr

# installation of fv3config
fv3config_sha1=1eb1f2898e9965ed7b32970bed83e64e074a7630
pip install git+git://github.com/VulcanClimateModeling/fv3config.git@${fv3config_sha1}

# installation of dace
export DACE_VERSION="linus-fixes-8"
rm -rf dace
git clone git@github.com:spcl/dace.git dace
cd dace
git checkout ${DACE_VERSION}
cd ../
pip install -e ./dace

# installation of gt4py
export GT4PY_VERSION=`cat GT4PY_VERSION.txt`
rm -rf gt4py
git clone git://github.com/gronerl/gt4py.git gt4py
cd gt4py
git checkout ${GT4PY_VERSION}
cd ../
pip install -e ./gt4py
python -m gt4py.gt_src_manager install

# installation of serialbox
rm -rf serialbox
git clone git@github.com:VulcanClimateModeling/serialbox.git serialbox
cd serialbox
mkdir build
cd build
cmake ../
cmake --build . -j 8
cmake --build . --target install
cd ..
export PYTHONPATH=`pwd`/install/python:$PYTHONPATH
cd ..


pip install ./external/fv3gfs-util/
pip install -e .
