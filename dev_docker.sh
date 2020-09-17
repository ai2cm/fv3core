#!/bin/bash

IMAGE=us.gcr.io/vcm-ml/fv3core-install:integration

MOUNTS="-v $(pwd):/fv3core"

#CONF_DIR=./lib/external/FV3/conf/

#cp $CONF_DIR/configure.fv3.gnu_docker $CONF_DIR/configure.fv3
docker run --rm $MOUNTS -w /fv3core -it $IMAGE bash
