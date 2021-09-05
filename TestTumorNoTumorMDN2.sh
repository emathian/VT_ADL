#!/bin/bash
#SBATCH --job-name=31TestTumorNoTumor_fromMNISTMDN2
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks-per-node=20      # nombre de taches MPI par noeud
#SBATCH --time=10:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=31TestTumorNoTumor_fromMNISTMDN2%j.out          # nom du fichier de sortie
#SBATCH --error=31TestTumorNoTumor_fromMNISTMDN2%j.out     
#SBATCH --account ohv@gpu
#SBATCH --gres=gpu:3 # 4
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
#python train.py
python test.py --model_path_AE /gpfswork/rech/ohv/ueu39kt/TumorNoTumor_fromMNISTMDN2/VT_AE_tumorNotumor_bs16.pt --model_path_MDN /gpfswork/rech/ohv/ueu39kt/TumorNoTumor_fromMNISTMDN2/G_estimate_tumorNotumor_bs16_.pt  --loss_outputpath /gpfsscratch/rech/ohv/ueu39kt/TumorNoTumor_fromMNISTMDN2_res/loss.csv --vector_outputpath /gpfsscratch/rech/ohv/ueu39kt/TumorNoTumor_fromMNISTMDN2_res/vector.csv --pi_outputpath  /gpfsscratch/rech/ohv/ueu39kt/TumorNoTumor_fromMNISTMDN2_res/pi.csv --mu_outputpath /gpfsscratch/rech/ohv/ueu39kt/TumorNoTumor_fromMNISTMDN2_res/mu.csv  --sigma_outputpath /gpfsscratch/rech/ohv/ueu39kt/TumorNoTumor_fromMNISTMDN2_res/sigma.csv --dirImg /gpfsscratch/rech/ohv/ueu39kt/TumorNoTumor_fromMNISTMDN2_res/outputImg