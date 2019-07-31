import sys
import argparse
import torch
import torchvision
import terrinus_dataset
import u_net
import softmax_nllloss
import os

def train_model(model, num_epochs, dataloaders, criterion, optimizer, use_gpu=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss = []
    val_loss = []
    for epoch in range(0,num_epochs):
        print("Epoch number {}/{}".format(epoch, num_epochs-1))
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
                print("Training****")
            else:
                print("Validation**")
                model.eval()
            
            loss = 0.0
            # Load data
            for inputs, labels in dataloaders[phase]:
                running_loss = 0.0
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.long()
                labels = labels.to(device)
                with torch.set_grad_enabled(phase=='train'):
                    out = model(inputs)
                    running_loss = criterion(out, labels)    
                    if(phase=='train'):
                        running_loss.backward()
                        optimizer.step()
                loss += running_loss
            print("Loss: {:.4f}".format(loss))  
            train_loss.append(loss)
            if(loss < max(train_loss)):
                torch.save(model.state_dict(), os.path.join(save_dir,"unet_{:.4f}".format(loss)))
                print("Saving model...")
# Define directories
train_dir_imgs = 'C:/Users/user/Ponc/terrinus/annotation/train_overfit/'
train_dir_masks = 'C:/Users/user/Ponc/terrinus/annotation/train_masks_overfit/'
val_dir_imgs = 'C:/Users/user/Ponc/terrinus/annotation/val_imgs_overfit/'
val_dir_masks = 'C:/Users/user/Ponc/terrinus/annotation/val_masks_overfit/'
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

dataset_val = terrinus.TerrinusDataset(val_dir_imgs, val_dir_masks)
dataloader_val = torch.utils.data.DataLoader(dataset_val,batch_size=2)

num_classes = 4
unet = u_net.UNet(num_classes = 4)
ex_batch = next(iter(dataloader_train))
out = unet(ex_batch[0])
# Compute LogSoftmax in the pixel (channel) dimension
soft = torch.nn.Softmax(dim=1)
nlll = torch.nn.NLLLoss()
lsm = soft(out)
print(lsm[0,:,0,0])
loss = nlll(lsm, ex_batch[1].long())
print(loss)
print(ex_batch[1].long()[0,0,0])
optimizer = torch.optim.Adam(unet.parameters(), lr=3e-4)

soft_nlll = softmax_nllloss.SoftMax_NLLL()
laloss = soft_nlll.forward(out, ex_batch[1].long())
print(laloss)
# 15,17,19,20,21,23,42,50
train_model(unet, 5, dataloader_train, soft_nlll, optimizer)