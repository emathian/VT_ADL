import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd
import seaborn as sns
import sklearn 
# import metric
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
import torch
import umap
from sklearn.preprocessing import StandardScaler
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--rootdir", required=True, default= '/gpfsscratch/rech/ohv/ueu39kt/TumorNoTumor_fromMNISNOTMDN2_res', help="Root directory of the experiment")
ap.add_argument("-v", "--vector_file_name", required=True, default= 'vector.csv', help="vector filename")
ap.add_argument("-pfn", "--plot_file_name", required=True, default= 'embedding_umap_vector.png', help="plot filename")
ap.add_argument("-oufn", "--output_umap_file_name", required=False, default= 'umap_emd01.csv', help="Output Loss filename")

args = vars(ap.parse_args())

rootdir = args['rootdir']
vectorf = args['vector_file_name']
plot_filename = args['plot_file_name'] 
umap_emd = args['output_umap_file_name']

df_vector = pd.read_csv(os.path.join(rootdir, vectorf ), header = None)
sample = []
for i in range(df_vector.shape[0]):
    sample.append(df_vector.iloc[i,0].split('/')[-3])
df_vector['sample'] = sample
df_vector_values = df_vector.iloc[:,1:-1]
scaled_dfvector_data = StandardScaler().fit_transform(df_vector_values)
scaled_dfvector_data = pd.DataFrame(scaled_dfvector_data)
reducer = umap.UMAP(n_components=3)
embedding = reducer.fit_transform(scaled_dfvector_data.values)
fig = plt.figure(figsize=(15, 9), dpi=150)
plt.scatter(
    embedding[:, 0],
    embedding[:, 1])
fig.savefig(os.path.join(rootdir, plot_filename), dpi=200)
plt.scatter(
    embedding[:, 0],
    embedding[:, 2])
fig.savefig(os.path.join(rootdir, 'umap_emd02.png'), dpi=200)

plt.scatter(
    embedding[:, 1],
    embedding[:, 2])
fig.savefig(os.path.join(rootdir, 'umap_emd12.png'), dpi=200)

embedding = pd.DataFrame(embedding)
embedding['filename'] = df_vector.iloc[:,0]
embedding.to_csv(os.path.join(rootdir, umap_emd))