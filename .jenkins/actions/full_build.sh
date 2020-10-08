#!/bin/bash
set -e
set -x
PULL=False make build
make test_gt4py
make tests
make tests_mpi
make push_environment
