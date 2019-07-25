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


def visualize_prediction(pred, color_code=[0,64,128,255]):
    # Input is 1*C*W*H, take the maximum and assign color code
    out_np = pred.detach().numpy()
    mask = np.zeros((np.shape(out_np)[2],np.shape(out_np)[3]))
    maxs = np.argmax(mask, axis=1)
    for i,c in enumerate(color_code):
        idxs = np.where(maxs==i)
        mask[idxs] = c
    return mask

# Define directories
train_dir_imgs = 'C:/Users/user/Ponc/terrinus/annotation/train_overfit/'
train_dir_masks = 'C:/Users/user/Ponc/terrinus/annotation/train_masks_overfit/'
path_to_pth = 'C:/Users/user/Ponc/terrinus/model_weights/unet_0.3520'
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
# load pretrained model
unet.load_state_dict(torch.load(path_to_pth))

out = unet(ex_batch[0])
logsoft = torch.nn.LogSoftmax(dim=1)
out = logsoft(out)
print(out.size())
out_np = out.detach().numpy()[0,:,:,:]
print(out_np.shape)
print(out_np[:,:10,:10])
out_np = np.amax(np.abs(out_np), axis=0)
print(out_np[:10,:10])
print(out_np.shape)
pred = visualize_prediction(out)
plt.imshow(out_np)
plt.show()