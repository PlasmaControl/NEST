#!/bin/bash
# Copy NEST states that live only on the Volume into the harvested output directory.
mkdir -p /work/out
cp /results/fxr/A60_cont/A60_cont_a0.1.npz /results/fxr/A60_cont/A60_cont_a0.3.npz /work/out/ 2>/dev/null
cp /results/fxr/A32_a03/final.npz /work/out/A32_a03_final.npz 2>/dev/null
cp /results/fxr/MQ2/MQ2_p16_a0.5.npz /work/out/ 2>/dev/null
cp /results/fxr/pre32/seed32_r0.6375_w0.07.npz /results/fxr/pre32/seed32_r0.9131_w0.03.npz /results/fxr/pre32/seed32_all.npz /work/out/ 2>/dev/null
cp /results/fxr/A60_cont/A60_cont.log /results/fxr/A60_cont/A60_cont.json /work/out/ 2>/dev/null
ls -la /work/out
