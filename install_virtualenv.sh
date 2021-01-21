#!/bin/bash
virtualenv_path=$1

root=`dirname $0`
(cd ${root}/external/daint_venv && ./install.sh ${virtualenv_path})
source ${virtualenv_path}/bin/activate
python3 -m pip install ${root}/external/fv3gfs-util/
python3 -m pip install -c ${root}/constraints.txt -r ${root}/requirements.txt
python3 -m pip install ${root}
deactivate
