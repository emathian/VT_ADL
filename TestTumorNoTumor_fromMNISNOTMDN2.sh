#!/bin/bash
#SBATCH --job-name=0408TumorNoTumor_fromMNISNOTMDN2
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-dev
#SBATCH --ntasks-per-node=20      # nombre de taches MPI par noeud
#SBATCH --time=01:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=0408TumorNoTumor_fromMNISNOTMDN2%j.out          # nom du fichier de sortie
#SBATCH --error=0408TumorNoTumor_fromMNISNOTMDN2%j.out     
#SBATCH --account ohv@gpu
#SBATCH --gres=gpu:3 # 4
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
#python train.py
python test.py --model_path_AE /gpfswork/rech/ohv/ueu39kt/TumorNoTumor_fromMNISNOTMDN2/VT_AE_tumorNotumor_bs16.pt   --loss_outputpath /gpfsscratch/rech/ohv/ueu39kt/TumorNoTumor_fromMNISNOTMDN2_res/loss_2.csv --vector_outputpath /gpfsscratch/rech/ohv/ueu39kt/TumorNoTumor_fromMNISNOTMDN2_res/vector_2.csv --dirImg /gpfsscratch/rech/ohv/ueu39kt/TumorNoTumor_fromMNISNOTMDN2_res/outputImg_2