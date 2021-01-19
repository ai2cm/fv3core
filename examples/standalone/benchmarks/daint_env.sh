#!/bin/bash
# get the root directory of fv3core
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPTPATH")")")"
echo $ROOT_DIR
cd $ROOT_DIR

cp -r /project/s1053/install/venv/vcm_1.0/ .
git submodule update --init --recursive

vcm_1.0/bin/python -m pip install external/fv3gfs-util/
vcm_1.0/bin/python -m pip install .
