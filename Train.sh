#!/bin/bash
#SBATCH --job-name=21TEstMNIST0
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks-per-node=20      # nombre de taches MPI par noeud
#SBATCH --time=10:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=21TestMNIST0%j.out          # nom du fichier de sortie
#SBATCH --error=21TestMNIST0%j.out     
#SBATCH --account ohv@gpu
#SBATCH --gres=gpu:3 # 4
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
#python train.py
python test.py
