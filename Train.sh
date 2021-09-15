#!/bin/bash
#SBATCH --job-name=DevMDN
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-dev
#SBATCH --ntasks-per-node=20      # nombre de taches MPI par noeud
#SBATCH --time=00:20:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=DevMDN%j.out          # nom du fichier de sortie
#SBATCH --error=DevMDN%j.out     
#SBATCH --account ohv@gpu
#SBATCH --gres=gpu:3 # 4
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
python train.py
# python test.py --model_path_AE /gpfswork/rech/ohv/ueu39kt/TumorNoTumor_fromMNISNOTMDN30/VT_AE_tumorNotumor_bs16.pt --loss_outputpath /gpfswork/rech/ohv/ueu39kt/TumorNoTumor_fromMNISNOTMDN30/loss.csv --vector_outputpath /gpfswork/rech/ohv/ueu39kt/TumorNoTumor_fromMNISNOTMDN30/vector.csv
