#!/bin/bash
#SBATCH --job-name=1509TestVTCTAT_
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks-per-node=20      # nombre de taches MPI par noeud
#SBATCH --time=100:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=1509TestVTCTAT_%j.out          # nom du fichier de sortie
#SBATCH --error=1509TestVTCTAT_%j.out     
#SBATCH --account ohv@gpu
#SBATCH --gres=gpu:3 # 4
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
#python train.py
python test.py --model_path_AE /gpfsscratch/rech/ohv/ueu39kt/TypicalAypical_fromMNISNOTMDN0609/VT_AE_tyical_atypical_bs16.pt   --loss_outputpath /gpfsscratch/rech/ohv/ueu39kt/TypicalAypical_fromMNISNOTMDN0609/res/loss_typical_test.csv --vector_outputpath /gpfsscratch/rech/ohv/ueu39kt/TypicalAypical_fromMNISNOTMDN0609/res/vector_typical_test.csv --img_path_list /gpfsscratch/rech/ohv/ueu39kt/Typical_Sample_CIRC_75_0.csv  --dirImg /gpfsscratch/rech/ohv/ueu39kt/TypicalAypical_fromMNISNOTMDN0609