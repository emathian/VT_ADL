#!/bin/bash
#SBATCH --job-name=ScratchTumorNoTumor
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=4      # nombre de taches MPI par noeud
#SBATCH --time=40:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=ScratchTumorNoTumor%j.out          # nom du fichier de sortie
#SBATCH --error=ScratchTumorNoTumor%j.out     
#SBATCH --account ohv@gpu
#SBATCH --gres=gpu:3
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
python train.py
#python testMNIST.py

