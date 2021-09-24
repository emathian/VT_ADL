# -*- coding: utf-8 -*-
"""
@author: Pankaj Mishra
"""

import torch
import mvtech
import torchvision.utils as utils
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import pytorch_ssim
import mdn1
from VT_AE import VT_AE as ae
import argparse

## Argparse declaration ##
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
        
    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 0 if not step else self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
ap = argparse.ArgumentParser()
ap.add_argument("-ts", "--trainset", required=False, default= '/gpfsscratch/rech/ohv/ueu39kt/Typical_Sample_Training.csv',
                help="Path to the traning set .csv list")
ap.add_argument("-s", "--summury_path", required=False, default= '/gpfsscratch/rech/ohv/ueu39kt/TypicalAypical_fromMNISNOTMDN0609', 
                help="Path to summary")
ap.add_argument("-chekVTAE", "--path_checkpoint_VTAE", required=False, default= '/gpfswork/rech/ohv/ueu39kt/TumorNoTumor_fromMNISTMDN2/VT_AE_tumorNotumor_bs16.pt', 
                help="Path to summary")
ap.add_argument("-chekGMM", "--path_checkpoint_GMM", required=False, default=   '/gpfswork/rech/ohv/ueu39kt/TumorNoTumor_fromMNISTMDN2/G_estimate_tumorNotumor_bs16_.pt', 
                help="Path to summary")
ap.add_argument("-MNVAE", "--model_name_VTADL", required=False, default= 'VT_AE_tyical_atypical_bs16.pt', 
                help="Path to summary")
ap.add_argument("-MNGMM", "--model_name_GMM", required=False, default= 'VT_AE_tyical_atypical_bs16.pt', 
                help="Path to summary")
ap.add_argument("-e", "--epochs", required=False, default= 100, help="Number of epochs to train")
ap.add_argument("-lr", "--learning_rate", required=False, default= 0.0001, help="learning rate")
ap.add_argument("-ps","--patch_size", required=False, default=64, help="Patch size of the images")
ap.add_argument("-dimVTADL","--dimVTADL", required=False, default=512, help="Number of dimension in transformer encoder")
ap.add_argument("-mdnc","--MDN_COEFS", required=False, default=150, help="Number of coef in the GMM")
ap.add_argument("-nh","--heads", required=False, default=16, help="Number of head in transformer")
ap.add_argument("-b", "--batch_size", required=False, default=8, help= "batch size")
ap.add_argument("-w", "--workers", required=False, default=4, help= "Nb process")
ap.add_argument("-gpu_ids", "--gpu_ids", required=False, default='0,1', help= "Nb gpus")

args = vars(ap.parse_args())

os.makedirs(args["summury_path"], exist_ok=True)

os.makedirs(os.path.join(args["summury_path"], 'runs'), exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(args["summury_path"], 'runs'))

epoch =args["epochs"]
minloss = 1e10
ep =0
ssim_loss = pytorch_ssim.SSIM() # SSIM Loss
#Dataset
train_dset = mvtech.Mvtec(root= args["trainset"])
train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=int(args["batch_size"]), shuffle=False,
        num_workers=int(args["workers"]), pin_memory=False)
# Model declaration
model = ae(patch_size=int(args["patch_size"]),depth=10, heads=int(args["heads"]), dim=int(args['dimVTADL']),train=True)

G_estimate= mdn1.MDN(coefs =int(args["MDN_COEFS"]))
use_cuda = torch.cuda.is_available()
if use_cuda:
    print( args['gpu_ids'].split(','))
    gpu_ids = list(map(int, args['gpu_ids'].split(',')))
    cuda='cuda:'+ str(gpu_ids[0])
    model = torch.nn.DataParallel(model,device_ids=gpu_ids)
    G_estimate = torch.nn.DataParallel(G_estimate,device_ids=gpu_ids)
device= torch.device(cuda if use_cuda else 'cpu')

if args['path_checkpoint_VTAE'] != 'None':
    model.load_state_dict(torch.load(args['path_checkpoint_VTAE']))

if args['path_checkpoint_GMM'] != 'None':
    G_estimate.load_state_dict(torch.load(args['path_checkpoint_GMM']))
    
  
model.to(device)
G_estimate.to(device)

# ### put model to train ##
#(The two models are trained as a separate module so that it would be easy to use as an independent module in different scenarios)
model.train()
G_estimate.train()

#Optimiser Declaration
encoder_embed_dim = 512
lr_factor = 2
lr_warmup = 4000
Optimiser = optimizer = NoamOpt(
    model_size=encoder_embed_dim, 
    factor=lr_factor, 
    warmup=lr_warmup, 
    optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))
#Optimiser = Adam(list(model.parameters())+list(G_estimate.parameters()), lr=args["learning_rate"], weight_decay=0.0001)

# ############## TRAIN #####################
# torch.autograd.set_detect_anomaly(True) #uncomment if you want to track an error

print('\nNetwork training started.....')
for i in range(epoch):
    t_loss = []

    for c, j in enumerate(train_loader):
        model.zero_grad()

        # vector,pi, mu, sigma, reconstructions = model(j.cuda())
        vector, reconstructions = model(j.cuda())
        pi, mu, sigma = G_estimate(vector)
        #Loss calculations
        loss1 = F.mse_loss(reconstructions, j.cuda(), reduction='mean') #Rec Loss
        loss2 = 1-ssim_loss(j.cuda(), reconstructions) #SSIM loss for structural similarity
        loss3 = mdn1.mdn_loss_function(vector,mu,sigma,pi) #MDN loss for gaussian approximation

        loss = 5*loss1 + 0.5*loss2 + loss3.sum()       #Total loss
        t_loss.append(loss.item())   #storing all batch losses to calculate mean epoch loss

        # Tensorboard definitions
        writer.add_scalar('recon-loss', loss1.item(), i*len(train_loader)* int(args["batch_size"]) + (c+1))
        writer.add_scalar('ssim loss', loss2.item(), i*len(train_loader)* int(args["batch_size"]) + (c+1))
        writer.add_scalar('Gaussian loss', loss3.item(), i*len(train_loader)* int(args["batch_size"]) + (c+1))
        writer.add_histogram('Vectors', vector)

        ## Uncomment below to store the distributions of pi, var and mean ##
#         writer.add_histogram('Pi', pi)
#         writer.add_histogram('Variance', sigma)
#         writer.add_histogram('Mean', mu)

        #Optimiser step
        loss.backward()

        Optimiser.step()
        if c%50000 == 0:
            #Tensorboard definitions for the mean epoch values
            writer.add_image('Reconstructed Image',utils.make_grid(reconstructions),i,dataformats = 'CHW')
            writer.add_scalar('Mean Epoch loss', np.mean(t_loss), i)
            print(f'Mean Epoch {i} loss: {np.mean(t_loss)}')
            print(f'Min loss epoch: {ep} with min loss: {minloss}')

            writer.close()

            # Saving the best model
            if np.mean(t_loss) <= minloss:
                minloss = np.mean(t_loss)
                ep = i
                os.makedirs(args["summury_path"], exist_ok=True)
                torch.save(model.state_dict(), 
                           os.path.join(args["summury_path"], args['model_name_VTADL']))
                torch.save(G_estimate.state_dict(),  os.path.join(args["summury_path"], args['model_name_GMM']))


'''
Full forms:
GN - gaussian Noise
LD = Linear Decoder
DR - Dynamic Routing
Gn = No of gaussian for the estimation of density, with n as the number
Pn = Pacth with n is dim of patch
SS - trained with ssim loss


'''
