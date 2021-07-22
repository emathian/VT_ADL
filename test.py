# -*- coding: utf-8 -*-
"""
@author: Pankaj Mishra
"""

import torch
import mvtech
import torch.nn.functional as F
import os
import numpy as np
import pytorch_ssim
from einops import rearrange
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import mdn1
from VT_AE import VT_AE as ae
from utility_fun import *
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", required=False, default= 400, help="Number of epochs to train")
ap.add_argument("-lr", "--learning_rate", required=False, default= 0.0001, help="learning rate")
ap.add_argument("-ps","--patch_size", required=False, default=64, help="Patch size of the images")
ap.add_argument("-b", "--batch_size", required=False, default=1, help= "batch size")
ap.add_argument("-w", "--workers", required=False, default=10, help= "Nb process")
ap.add_argument("-gpu_ids", "--gpu_ids", required=False, default='0,1,2', help= "Nb gpus")

args = vars(ap.parse_args())


prdt = "hazelnut"
patch_size = 64

ssim_loss = pytorch_ssim.SSIM() # SSIM Loss

model = ae(patch_size=args["patch_size"],depth=10, heads=16,train=False)
G_estimate= mdn1.MDN()
use_cuda = torch.cuda.is_available()
if use_cuda:
    print( args['gpu_ids'].split(','))
    gpu_ids = list(map(int, args['gpu_ids'].split(',')))
    cuda='cuda:'+ str(gpu_ids[0])
    model = torch.nn.DataParallel(model,device_ids=gpu_ids)
    G_estimate = torch.nn.DataParallel(G_estimate,device_ids=gpu_ids)
device= torch.device(cuda if use_cuda else 'cpu')

model.load_state_dict(torch.load('/gpfsscratch/rech/ohv/ueu39kt/MNIST_Norm0_MDN/VT_AE_MNIST0_bs16.pt'))
G_estimate.load_state_dict(torch.load('/gpfsscratch/rech/ohv/ueu39kt/MNIST_Norm0_MDN/G_estimate_MNIST0_bs16_.pt'))
model.to(device)
G_estimate.to(device)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# model.cuda()
#put model to eval
model.eval()
G_estimate.eval()

train_dset = mvtech.Mvtec(train=False)
train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args["batch_size"], shuffle=True,
        num_workers=args["workers"], pin_memory=False)
#### testing #####
loader = [train_loader]

t_loss_norm =[]
t_loss_anom =[]



def Patch_Overlap_Score(data_load = loader[:1],  upsample =1, out_file = '/gpfsscratch/rech/ohv/ueu39kt/MNIST_Norm0_MDN/loss_MNIST0.csv', out_file2 = '/gpfsscratch/rech/ohv/ueu39kt/MNIST_Norm0_MDN/vectors_MNIST0.csv'):
    
    norm_loss_t = []
    normalised_score_t = []
    mask_score_t = []


    labels_l = []
    
    score_tn = []
    score_ta = []
    
    vectors = []
    for n,data in enumerate(data_load):
        total_loss_all = []
        for c,(img, label) in enumerate(data):
            vector, reconstructions = model(img.cuda())
            pi, mu, sigma = G_estimate(vector)
            #Loss calculations
            loss1 = F.mse_loss(reconstructions,img.cuda(), reduction='mean') #Rec Loss
            loss2 = 1-ssim_loss(img.cuda(), reconstructions) #SSIM loss for structural similarity
            loss3 = mdn1.mdn_loss_function(vector,mu,sigma,pi, test= True) #MDN loss for gaussian approximation
            loss = 5*loss1 + 0.5*loss2 + loss3.max()       #Total loss
            total_loss_all.append(loss.detach().cpu().numpy())
            labels_l.append(label)
            vectors.append(vector.detach().cpu().numpy())
            # Plotting
            if c%5 == 0:
                img = img.detach().cpu().numpy()
                reconstructions = reconstructions.detach().cpu().numpy()
                plot(img,reconstructions, 'img_{}_{}_{}.png'.format(str(label[0]), str(c), str(n)))
            print(' label ', label[0])
            loss1 = loss1.detach().cpu().numpy()
            loss2 =  loss2.detach().cpu().numpy()
            loss3 = loss3.max().detach().cpu().numpy()
            print('  loss1 ', loss1, type(loss1))
            print('  loss2  ', loss2, type(loss2))
            print('  loss3  ', loss3, type(loss3))
            print('  vector  ', vector, type(vector), vector.size())

            with open(out_file, 'a') as f1:
                for ii in range(len(label)):
                    f1.write('{}\t{}\t{}\t{}\n'.format(label[0], loss1, loss2, loss3))
            f1.close()
            with open(out_file2, 'a') as f2:
                f2.write('{}\t{}\n'.format(label[0], vector.detach().cpu().numpy().flatten()))
            f2.close()
            vectors = []
    
    return -1
if __name__=="__main__":
    
    #thres = Thresholding()
    Patch_Overlap_Score()


