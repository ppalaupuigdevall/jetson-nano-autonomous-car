import sys
import argparse
import torch
import torchvision
import terrinus_dataset
import u_net
import softmax_nllloss
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_prediction(pred, color_code=[0,64,128,255]):
    # Input is 1*C*W*H, take the maximum and assign color code
    # sm = torch.nn.LogSoftmax(dim=1)
    # pred = sm(pred)
    out_np = pred.detach().numpy()
    print("out_np.shape")
    print(out_np.shape)
    
    mask = np.zeros((np.shape(out_np)[2],np.shape(out_np)[3]))
    maxs = np.argmax(out_np, axis=1)
    print("Maxs shape")
    print(maxs.shape)
    print(maxs[:4,:4])
    for i,c in enumerate(color_code):
        idxs = np.where(maxs[0,:,:]==i)
        mask[idxs] = c
    return mask

# Define directories
train_dir_imgs = 'C:/Users/user/Ponc/terrinus/annotation/train_overfit/'
train_dir_masks = 'C:/Users/user/Ponc/terrinus/annotation/train_masks_overfit/'
path_to_pth = 'C:/Users/user/Ponc/terrinus/model_weights/unet_0.9509_85'
# path_to_pth = 'C:/Users/user/Ponc/terrinus/model_weights/unet_0.3520'
path_to_pth = 'C:/Users/user/Ponc/terrinus/model_weights/unet_0.1740_85'

# Define transforms
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = torchvision.transforms.Compose([
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize(*mean_std)
])
# Define dataset and dataloader
dataset_train = terrinus_dataset.TerrinusDataset(train_dir_imgs, train_dir_masks, transforms=input_transform)
bs = 1
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = bs)

ex_batch = next(iter(dataloader_train))
num_classes = 4

unet = u_net.UNet(num_classes = 4)
# load pretrained model
unet.load_state_dict(torch.load(path_to_pth, map_location='cpu'))

out = unet(ex_batch[0])
logsoft = torch.nn.LogSoftmax(dim=1)
out = logsoft(out)
# print(out.size())
out_np = out.detach().numpy()[0,:,:,:]
pred = visualize_prediction(out)
# plt.imshow(pred)
# plt.show()

pred = np.uint8(pred)
cv2.imshow('prediction', pred)
cv2.waitKey(0)

kernel = np.ones((9,9),np.uint8)
opening = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)
cv2.waitKey(0)
plt.subplot(141)
plt.imshow(out_np[0,:,:])
plt.subplot(142)
plt.imshow(out_np[1,:,:])
plt.subplot(143)
plt.imshow(out_np[2,:,:])
plt.subplot(144)
plt.imshow(out_np[3,:,:])
plt.show()
laplacian = cv2.Laplacian(opening,cv2.CV_64F)
laplacian = np.uint8(laplacian)
cv2.imshow('laplsacian',laplacian)
cv2.waitKey(0)
ret,thresh1 = cv2.threshold(laplacian,127,255,cv2.THRESH_BINARY)

cv2.imshow('binary',thresh1)
cv2.waitKey(0)
oness = np.where(thresh1 == 255)
# print(oness)
mean_pix_x = np.average(oness[0])
mean_pix_y = np.average(oness[1])
# print(mean_pix_x)
# print(mean_pix_y)
cv2.circle(pred,(np.int32(mean_pix_x),(np.int32(mean_pix_y))), 4, (0,0,255), -1)
cv2.imshow('ei',pred)
cv2.waitKey(0)