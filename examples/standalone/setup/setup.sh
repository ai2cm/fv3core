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

# install of boost
wget -q https://boostorg.jfrog.io/artifactory/main/release/1.74.0/source/boost_1_74_0.tar.gz
tar xzf boost_1_74_0.tar.gz
cd boost_1_74_0
mkdir include
cp -r boost include
BOOST_ROOT=`pwd`/boost_1_74_0

# installation of gt4py
export GT4PY_VERSION=`cat GT4PY_VERSION.txt`
rm -rf gt4py
git clone git://github.com/VulcanClimateModeling/gt4py.git gt4py
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
