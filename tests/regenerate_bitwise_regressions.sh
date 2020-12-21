#!/bin/bash
cd ../

make tests_mpi TEST_ARGS="--which_modules=FVDynamics,FVSubgridZ --force-regen"
sudo chown -R $USER:$USER test_data/${EXPERIMENT}
# To make this permanent, run :
#  EXPERIMENT=<experiment>  make push_python_regressions
