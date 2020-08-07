#!/bin/bash
set -e
set -x
ARGS=$1
make tests TEST_ARGS="$ARGS"
make tests_mpi TEST_ARGS="$ARGS"
