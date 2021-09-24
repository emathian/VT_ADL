#!/bin/bash
#SBATCH --job-name=TestACExp1
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-dev
#SBATCH --ntasks-per-node=20      # nombre de taches MPI par noeud
#SBATCH --time=00:20:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=TestACExp1_%j.out          # nom du fichier de sortie
#SBATCH --error=TestACExp1_%j.out     
#SBATCH --account ohv@gpu
#SBATCH --gres=gpu:3 # 4
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
python test.py --testset /gpfsscratch/rech/ohv/ueu39kt/PathListVTADL_EXP1/Atypical_Sample_test.csv \
                --path_checkpoint_VTAE /gpfsscratch/rech/ohv/ueu39kt/TypicalAypical_Exp1/VT_AE_tyical_atypical_exp1.pt \
                --path_checkpoint_GMM 'None' \
                --patch_size 32 \
                --dimVTADL 1024 \
                --MDN_COEFS 300 \
                --heads 16 \
                --batch_size 4 \
                --workers 4 \
                --gpu_ids 0,1,2 \
                --loss_outputpath /gpfsscratch/rech/ohv/ueu39kt/TypicalAypical_Exp1/res/loss_AC.csv \
                --vector_outputpath /gpfsscratch/rech/ohv/ueu39kt/TypicalAypical_Exp1/res/vector_AC.csv \
                --dirImg /gpfsscratch/rech/ohv/ueu39kt/TypicalAypical_Exp1/ReconstrcutedImg