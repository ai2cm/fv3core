#!/bin/sh
sudo /opt/nvidia/nsight-compute/2021.1.0/ncu -f -o results.nvprof \
    --target-processes all --import-source on \
    /home/floriand/venv/vcm_1_0/bin/python3 \
    repro_compute_x_flux_2021-05-10_14-30-55/m_compute_x_flux__gtcuda_1bdfdf5c50/repro.py