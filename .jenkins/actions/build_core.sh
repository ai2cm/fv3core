#!/bin/bash
set -e -x
export DOCKER_BUILDKIT=1
make build  
make push_core
make tar_core
