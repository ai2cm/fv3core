#!/bin/bash
set -e
set -x
PULL=False make tests
PULL=False make tests_mpi
make push_environment
make push_core
make tar_core
