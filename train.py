import sys
import argparse
import torch
import torchvision
import terrinus
import u_net
# Define directories
train_dir_imgs = 'C:/Users/user/Ponc/terrinus/annotation/train_overfit/'
train_dir_masks = 'C:/Users/user/Ponc/terrinus/annotation/train_masks_overfit/'

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
s = torch.nn.Softmax2d()
nlll = torch.nn.NLLLoss()
sm = s(out)
loss = nlll(sm, ex_batch[1].long())
print(loss)

def train_model(model, num_epochs, dataloader_train, criterion, optimizer, dataloader_val=None):
    train_loss = []
    for epoch in num_epochs:
        print("Epoch number {}/{}".format(epoch, num_epochs))
        # Load data
        for inputs, labels in dataloader_train:
            print()