#!/bin/bash
#SBATCH --job-name=UMAPNoMDN
## SBATCH -C v100-32g
## SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks-per-node=20      # nombre de taches MPI par noeud
#SBATCH --time=20:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=UMAPNoMDN%j.out          # nom du fichier de sortie
#SBATCH --error=UMAPNoMDN%j.out     
#SBATCH --account ohv@cpu
##SBATCH --gres=gpu:3 # 4
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
#python train.py
python UMAP_vector.py --rootdir /gpfsscratch/rech/ohv/ueu39kt/TumorNoTumor_fromMNISNOTMDN2_res --vector_file_name vector.csv --plot_file_name embedding_umap_vector.png --output_umap_file_name umap_emd.csv