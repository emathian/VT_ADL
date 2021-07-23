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
ap.add_argument("-e", "--epochs", required=False, default= 100, help="Number of epochs to train")
ap.add_argument("-lr", "--learning_rate", required=False, default= 0.0001, help="learning rate")
ap.add_argument("-ps","--patch_size", required=False, default=64, help="Patch size of the images")
ap.add_argument("-b", "--batch_size", required=False, default=8, help= "batch size")
ap.add_argument("-w", "--workers", required=False, default=4, help= "Nb process")
ap.add_argument("-gpu_ids", "--gpu_ids", required=False, default='0,1', help= "Nb gpus")

args = vars(ap.parse_args())

writer = SummaryWriter()

epoch =args["epochs"]
minloss = 1e10
ep =0
ssim_loss = pytorch_ssim.SSIM() # SSIM Loss

#Dataset
train_dset = mvtech.Mvtec()
train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args["batch_size"], shuffle=False,
        num_workers=args["workers"], pin_memory=False)
# Model declaration
model = ae(patch_size=args["patch_size"],depth=10, heads=16,train=True)
G_estimate= mdn1.MDN()
use_cuda = torch.cuda.is_available()
if use_cuda:
    print( args['gpu_ids'].split(','))
    gpu_ids = list(map(int, args['gpu_ids'].split(',')))
    cuda='cuda:'+ str(gpu_ids[0])
    model = torch.nn.DataParallel(model,device_ids=gpu_ids)
    G_estimate = torch.nn.DataParallel(G_estimate,device_ids=gpu_ids)
device= torch.device(cuda if use_cuda else 'cpu')


#model.load_state_dict(torch.load('/gpfsscratch/rech/ohv/ueu39kt/MNIST_Norm0/VT_AE_MNIST0_bs16.pt'))

print('Model loaded :)')
#G_estimate.load_state_dict(torch.load('/gpfsscratch/rech/ohv/ueu39kt/saved_model_bs16_sample_1207/G_estimate_Mvtech_bs16_.pt'))

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
        #print(pi, mu, sigma)
        #Loss calculations
        loss1 = F.mse_loss(reconstructions, j.cuda(), reduction='mean') #Rec Loss
        loss2 = 1-ssim_loss(j.cuda(), reconstructions) #SSIM loss for structural similarity
        loss3 = mdn1.mdn_loss_function(vector,mu,sigma,pi) #MDN loss for gaussian approximation

        loss = 5*loss1 + 0.5*loss2 + loss3       #Total loss
        print('Loss ', loss.item())
        t_loss.append(loss.item())   #storing all batch losses to calculate mean epoch loss

        # Tensorboard definitions
        writer.add_scalar('recon-loss', loss1.item(), i*len(train_loader)* args["batch_size"] + (c+1))
        writer.add_scalar('ssim loss', loss2.item(), i*len(train_loader)* args["batch_size"] + (c+1))
        writer.add_scalar('Gaussian loss', loss3.item(), i)
        writer.add_histogram('Vectors', vector)

        ## Uncomment below to store the distributions of pi, var and mean ##
        writer.add_histogram('Pi', pi)
        writer.add_histogram('Variance', sigma)
        writer.add_histogram('Mean', mu)

        #Optimiser step
        loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
#         torch.nn.utils.clip_grad_norm_(G_estimate.parameters(), max_norm=2.0, norm_type=2)

        Optimiser.step()

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
        os.makedirs('/gpfsscratch/rech/ohv/ueu39kt/MNIST_Norm0_MDN', exist_ok=True)
        torch.save(model.state_dict(), f'/gpfsscratch/rech/ohv/ueu39kt/MNIST_Norm0_MDN/VT_AE_MNIST0_bs16'+'.pt')
        torch.save(G_estimate.state_dict(), f'/gpfsscratch/rech/ohv/ueu39kt/MNIST_Norm0_MDN/G_estimate_MNIST0_bs16_'+'.pt')


'''
Full forms:
GN - gaussian Noise
LD = Linear Decoder
DR - Dynamic Routing
Gn = No of gaussian for the estimation of density, with n as the number
Pn = Pacth with n is dim of patch
SS - trained with ssim loss


'''
