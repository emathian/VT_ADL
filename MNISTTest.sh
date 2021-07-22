#!/bin/bash
#SBATCH --job-name=LNENPretrain2
##SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=8        # nombre de taches MPI par noeud
#SBATCH --time=00:30:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=2LNENPretraining%j.out          # nom du fichier de sortie
#SBATCH --error=2LNENPretraining%j.out     
#SBATCH --account ohv@gpu
#SBATCH --gres=gpu:1
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
python testMNIST.py
