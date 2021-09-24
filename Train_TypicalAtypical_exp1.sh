#!/bin/bash
#SBATCH --job-name=LoadTest
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-dev
#SBATCH --ntasks-per-node=20      # nombre de taches MPI par noeud
#SBATCH --time=00:20:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=LoadTest_%j.out          # nom du fichier de sortie
#SBATCH --error=LoadTest_%j.out     
#SBATCH --account ohv@gpu
#SBATCH --gres=gpu:3 # 4
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
python train.py --trainset /gpfsscratch/rech/ohv/ueu39kt/PathListVTADL_EXP1/Typical_Sample_Train.csv \
                --path_checkpoint_VTAE /gpfsscratch/rech/ohv/ueu39kt/TypicalAypical_Exp1/VT_AE_tyical_atypical_exp1.pt \
                --summury_path /gpfsscratch/rech/ohv/ueu39kt/TypicalAypical_Exp1 \
                --path_checkpoint_VTAE 'None' \
                --path_checkpoint_GMM 'None' \
                --model_name_VTADL VT_AE_tyical_atypical_exp1.pt \
                --model_name_GMM GMM_tyical_atypical_exp1.pt \
                --learning_rate 0.0001\
                --patch_size 32 \
                --dimVTADL 1024 \
                --MDN_COEFS 300 \
                --heads 16 \
                --batch_size 4 \
                --workers 4 \
                --gpu_ids 0,1,2