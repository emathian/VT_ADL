# -*- coding: utf-8 -*-
"""
@author: Pankaj Mishra
"""
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from collections import OrderedDict
from itertools import chain
import torch.utils.data as data
import random
from PIL import Image
random.seed(123)


def add_noise(latent, noise_type="gaussian", sd=0.2):
    """Here we add noise to the latent features concatenated from the 4 autoencoders.
    Arguements:
    'gaussian' (string): Gaussian-distributed additive noise.
    'speckle' (string) : Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.
    'sd' (integer) : standard deviation used for geenrating noise

    Input :
        latent : numpy array or cuda tensor.

    Output:
        Array: Noise added input, can be np array or cuda tnesor.
    """
    assert sd >= 0.0
    if noise_type == "gaussian":
        mean = 0.

        n = torch.distributions.Normal(torch.tensor([mean]), torch.tensor([sd]))
        noise = n.sample(latent.size()).squeeze(-1)#.cuda()
        latent = latent + noise
        return latent

    if noise_type == "speckle":
        noise = torch.randn(latent.size())#.cuda()
        latent = latent + latent * noise
        return latent

class Mvtec(data.Dataset):
    def __init__(self, root="/linkhome/rech/genkmw01/ueu39kt/osutils/MNIST.txt"):
        self.root = root
        torch.manual_seed(123)
        with open(root, 'r') as f:
            content =  f.readlines()
        self.files_list = []
        for x in content:
            x =  x.strip()
            if x.find('reject') == -1:
                self.files_list.append(x)

        ## Image Transformation ##
        # High color augmntation
        # Random orientation
        self.transform = transforms.Compose([
            transforms.Resize((550,550)),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            transforms.ToTensor(),
#            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def __getitem__(self,index):
        img =  Image.open(self.files_list[index])
        w, h = img.size
        ima = Image.new('RGB', (w,h))
        data = zip(img.getdata(), img.getdata(), img.getdata())
        ima.putdata(list(data))
        if self.transform is not None:
            img_c = self.transform(ima)
        img_n = add_noise(img_c)
        return (img_n, img_c)

    def __len__(self):
        return len(self.files_list)


if __name__ == "__main__":

    # print('======== All Normal Data ============')
    # Train_data(root, 'all')
    # print('======== All Anomaly Data ============')
    # Test_anom_data(root,'all')
	batch_size = 1
	trainds = Mvtec()
	train_loader  = torch.utils.data.DataLoader(trainds, batch_size=batch_size, shuffle=True)
	for i, j in train_loader:
		i = i[0,:,:,:]
		i = i.numpy()
		b  = (i - np.min(i)) /  np.ptp(i)

		plt.imshow(b.transpose(1,2,0))
		plt.savefig('original.png')
		break
