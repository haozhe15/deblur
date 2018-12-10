#!/bin/bash
#SBATCH --mem=10G
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH -p gpu-common --gres=gpu:1
#SBATCH -c 4
singularity exec --nv docker://capjon/tfgpu bash ~/deblur/run.sh
