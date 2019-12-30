import torch
import numpy as np
import cv2
import time
from torchvision import transforms
from PIL import Image
from dataset import imresize
from torch2trt import torch2trt
from models.models import MobileNetV2Dilated
from models.models import C1DeepSup
import models.mobilenet as mobilenet
from terri_utils import img_transform
from terri_utils import round2nearest_multiple


torch.cuda.set_device(0)

# LOAD IMAGE
imgMaxSize = 1000

image_path = "/home/dlinano/territest_22.jpeg"

img = Image.open(image_path).convert('RGB')
ori_width, ori_height = img.size

# image transform, to torch float tensor 3xHxW
img = img_transform(img)
img = torch.unsqueeze(img, 0)
print("img size = " + str(img.size()))
with torch.no_grad():
    orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=False)
    net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
    enc_weights = "/home/dlinano/maweights/ade20k-mobilenetv2dilated-c1_deepsup/encoder_epoch_20.pth"
    net_encoder.load_state_dict(torch.load(enc_weights, map_location='cuda:0'), strict=False)
    
    net_decoder = C1DeepSup(
                num_class=150,
                fc_dim=320,
                use_softmax=True)

    dec_weights = "/home/dlinano/maweights/ade20k-mobilenetv2dilated-c1_deepsup/decoder_epoch_20.pth"
    net_decoder.load_state_dict(
                torch.load(dec_weights, map_location='cuda:0'), strict=False)
    net_encoder = net_encoder.cuda(0)
    net_decoder = net_decoder.cuda(0)
    print("Models creats")
    for i in range(0,10):
        start_time = time.time()
        #net_encoder_trt = torch2trt(net_encoder, [torch.ones((1,3,480,640)).cuda(0)])
        #auxi = net_encoder_trt(img.cuda(0))

        auxi = net_encoder(img.cuda(0))
        #end_time = time.time()
    
        #print("Time ENCODING = " + str(end_time - start_time))
        #start_time = time.time()
        pred = net_decoder(auxi, segSize=(480, 640))
        end_time = time.time()
        print("Time DECODING = " + str(end_time - start_time))

