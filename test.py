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
ap.add_argument("-w", "--workers", required=False, default=1, help= "Nb process")
ap.add_argument("-gpu_ids", "--gpu_ids", required=False, default='0,1,2', help= "Nb gpus")
ap.add_argument("-mpAE", "--model_path_AE", required=True, default='/gpfswork/rech/ohv/ueu39kt/TumorNoTumor_fromMNISNOTMDN30/VT_AE_tumorNotumor_bs16.pt', help= "Model AE path")
ap.add_argument("-mpMDN", "--model_path_MDN", required=False, default=None, help= "Model MDN path")
ap.add_argument("-lout", "--loss_outputpath", required=True, default=None, help= "Output path of the loss.csv file")
ap.add_argument("-vout", "--vector_outputpath", required=True, default=None, help= "Output path of the vector.csv file")
ap.add_argument("-piout", "--pi_outputpath", required=False, default=None, help= "Output path of the pi.csv file")
ap.add_argument("-muout", "--mu_outputpath", required=False, default=None, help= "Output path of the mu.csv file")
ap.add_argument("-sigmaout", "--sigma_outputpath", required=False, default=None, help= "Output path of the sigma.csv file")
ap.add_argument("-dImg", "--dirImg", required=False, default=None, help= "Output path of reconstructes img file")

args = vars(ap.parse_args())


patch_size = 64

ssim_loss = pytorch_ssim.SSIM() # SSIM Loss

model = ae(patch_size=args["patch_size"],depth=10, heads=16,train=False)
if args['model_path_MDN']:
    G_estimate= mdn1.MDN()
use_cuda = torch.cuda.is_available()
if use_cuda:
    print( args['gpu_ids'].split(','))
    gpu_ids = list(map(int, args['gpu_ids'].split(',')))
    cuda='cuda:'+ str(gpu_ids[0])
    model = torch.nn.DataParallel(model,device_ids=gpu_ids)
    if args['model_path_MDN']:
        G_estimate = torch.nn.DataParallel(G_estimate,device_ids=gpu_ids)
device= torch.device(cuda if use_cuda else 'cpu')
model.load_state_dict(torch.load(args['model_path_AE']))
if args['model_path_MDN']:
    G_estimate.load_state_dict(torch.load(args['model_path_MDN']))
model.to(device)
if args['model_path_MDN']:
    G_estimate.to(device)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#put model to eval
model.eval()
if args['model_path_MDN']:
    G_estimate.eval()

train_dset = mvtech.Mvtec(test=True)
train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args["batch_size"], shuffle=False,
        num_workers=args["workers"], pin_memory=False)
#### testing #####
loader = [train_loader]
t_loss_norm =[]
t_loss_anom =[]



def Patch_Overlap_Score(data_load = loader[:1],  upsample =1, out_file1 = args['loss_outputpath'], out_file2 =args['vector_outputpath'], MDNPath = args['model_path_MDN'], out_file_pi = args['pi_outputpath'], out_file_mu =  args['mu_outputpath'], out_file_sigma =  args['sigma_outputpath'], dirImg  = args['dirImg']):
    
    norm_loss_t = []
    normalised_score_t = []
    mask_score_t = []


    labels_l = []
    score_tn = []
    score_ta = []
    vectors = []
    total_loss_all = []
    tiles_analysed = []
    with torch.no_grad():
        for c,(ele) in enumerate(data_load):
            print('ele ' , ele)
            img, label = ele
            label = label[0]
            if label not in tiles_analysed:
                tiles_analysed.append(label)
                vector, reconstructions = model(img.cuda())
                if MDNPath:
                    pi, mu, sigma = G_estimate(vector)
                #Loss calculations
                loss1 = F.mse_loss(reconstructions,img.cuda(), reduction='mean') #Rec Loss
                loss2 = 1-ssim_loss(img.cuda(), reconstructions) #SSIM loss for structural similarity
                if MDNPath:
                    loss3 = mdn1.mdn_loss_function(vector,mu,sigma,pi, test= True) #MDN loss for gaussian approximation
                    loss = 5*loss1 + 0.5*loss2 + loss3.max()       #Total loss
                else:
                    loss = 5*loss1 + 0.5*loss2 #+ loss3.max()
                total_loss_all.append(loss.detach().cpu().numpy())
                labels_l.append(label)
                vectors.append(vector.detach().cpu().numpy())
                # Plotting
                if c%10 == 0:
                    img = img.detach().cpu().numpy()
                    reconstructions = reconstructions.detach().cpu().numpy()
                    filename = str(label).split('.')[0]
                    plot(img,reconstructions, '{}.png'.format(str(filename), str(c), str(n)), dirImg = dirImg)
        #             print(' label ', label[0])
                loss1 = loss1.detach().cpu().numpy()
                loss2 =  loss2.detach().cpu().numpy()
                if MDNPath:
                    loss3 = loss3.max().detach().cpu().numpy()

                with open(out_file1, 'a') as f1:
                    for ii in range(len(label)):
                        if MDNPath:
                            f1.write('{}\t{}\t{}\t{}\n'.format(label, loss1, loss2, loss3))
                        else:
                            f1.write('{}\t{}\t{}\n'.format(label, loss1, loss2))
                f1.close()
                vector =  vector.cuda(0)
                Vector = vector.detach().cpu().numpy().flatten()
                Vector = np.random.choice(Vector,1000)
                Vector = Vector.reshape(1,Vector.shape[0])

                c_pd =  pd.DataFrame(data=Vector, index=[label])
                c_pd.to_csv(out_file2, mode='a', header=False)

                if MDNPath:
                    Pi = pi.detach().cpu().numpy().flatten()
                    Pi = np.random.choice(Pi,1000)
                    Pi = Pi.reshape(1,Pi.shape[0])


                    c_pd =  pd.DataFrame(data=Pi, index=[label])
                    c_pd.to_csv(out_file_pi, mode='a', header=False)

                    Mu = mu.detach().cpu().numpy().flatten()
                    Mu = np.random.choice(Mu,1000)
                    Mu = Mu.reshape(1,Mu.shape[0])

                    c_pd =  pd.DataFrame(data=Mu, index=[label])
                    c_pd.to_csv(out_file_mu, mode='a', header=False)


                    Sigma = sigma.detach().cpu().numpy().flatten()
                    Sigma = np.random.choice(Sigma,1000)
                    Sigma = Sigma.reshape(1,Sigma.shape[0])


                    c_pd =  pd.DataFrame(data=Sigma, index=[label])
                    c_pd.to_csv(out_file_sigma, mode='a', header=False)
            
            else:
                print(label)
                with open(out_file3, 'a') as f1:
                    for ii in range(len(label)):
                        f1.write('{}\t{}\t{}\n'.format(label, c))
                f1.close()
            del label, vector, reconstructions, loss1, loss2, Vector, c_pd, loss
            if MDNPath:
                del  Mu, Sigma, Pi, mu, sigma, pi, loss3, loss



    
    return -1
if __name__=="__main__":
    
    #thres = Thresholding()
    Patch_Overlap_Score()


