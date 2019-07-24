import sys
import argparse
import torch
import torchvision
import terrinus
import u_net
import softmax_nllloss
import os
import matplotlib.pyplot as plt
import numpy as np
def visualize_prediction(pred, color_code):
    # Input is 1*W*H*C, take the maximum and assign color code
    out_np = pred.numpy()
    mask = np.zeros()

# Define directories
train_dir_imgs = 'C:/Users/user/Ponc/terrinus/annotation/train_overfit/'
train_dir_masks = 'C:/Users/user/Ponc/terrinus/annotation/train_masks_overfit/'
save_dir = 'C:/Users/user/Ponc/terrinus/model_weights/'
# Define transforms
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = torchvision.transforms.Compose([
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(*mean_std)
])
# Define dataset and dataloader
dataset_train = terrinus.TerrinusDataset(train_dir_imgs, train_dir_masks, transforms=input_transform)
bs = 1
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = bs)

ex_batch = next(iter(dataloader_train))
num_classes = 4
unet = u_net.UNet(num_classes = 4)
out = unet(ex_batch[0])
# Compute LogSoftmax in the pixel (channel) dimension
logsoft = torch.nn.LogSoftmax(dim=1)
nlll = torch.nn.NLLLoss()
lsm = logsoft(out)
print(lsm[0,:,0,0])
loss = nlll(lsm, ex_batch[1].long())
print(loss)
print(ex_batch[1].long()[0,0,0])
optimizer = torch.optim.Adam(unet.parameters(), lr=3e-4)

soft_nlll = softmax_nllloss.SoftMax_NLLL()
laloss = soft_nlll.forward(out, ex_batch[1].long())
print(laloss)

train_model(unet, 20, dataloader_train, soft_nlll, optimizer)