#!/bin/bash
echo "task $SLURM_PROCID host $(hostname) CVD=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,pci.bus_id,compute_mode,memory.used --format=csv,noheader
/scratch/gpfs/EKOLEMEN/lk1158/envs/nest_mg/bin/python -c "import jax; print('task', $SLURM_PROCID, jax.devices())" 2>&1 | grep -v cpu_aot | tail -2
