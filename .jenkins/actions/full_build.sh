#!/bin/bash
# Builds and tests an image using the current gt4py develop branch
# instead of the pinned version in requirements.txt

set -e
set -x
make pull_core test_gt4py_develop
