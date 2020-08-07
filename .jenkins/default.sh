#!/bin/bash
set -e
set -x
make tests TEST_ARGS="$ARGS"
make tests_mpi TEST_ARGS="$ARGS"
