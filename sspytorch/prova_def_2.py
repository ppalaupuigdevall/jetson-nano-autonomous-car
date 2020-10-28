import torch
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from dataset import imresize
from models.models import MobileNetV2Dilated
from models.models import C1DeepSup
import models.mobilenet as mobilenet
from terri_utils import img_transform
from terri_utils import round2nearest_multiple
from lib.utils import as_numpy
# For DC motor
from adafruit_servokit import ServoKit # seteja el pinout a 1000 (GPIO.TEGRA_SOC)
import RPi.GPIO as GPIO
from camera import Camera
from processing_unit import GeomPU
# For csi camera
from jetcam.csi_camera import CSICamera
# For Servo motor
import board
import busio


# HW
output_pin = "GPIO_PE6"
if output_pin is None:
    raise Exception('PWM not supported on this board')

in1_pin = "UART2_RTS"
in2_pin = "DAP4_SCLK"

# Pin Setup:
# Board pin-numbering scheme
#GPIO.setmode(GPIO.BOARD)
def set_movement(in1, in2, direction='f'):
    if direction == 'f':
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
    else:
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)

# set pin as an output pin with optional initial state of HIGH
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)
p = GPIO.PWM(output_pin,50)
val = 0
p.start(val)
# Configure gpio pins 11 12
GPIO.setup(in1_pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(in2_pin, GPIO.OUT, initial=GPIO.HIGH)
#print("PWM running. Press CTRL+C to exit.")

# On the Jetson Nano
# Bus 0 (pins 28,27) is board SCL_1, SDA_1 in the jetson board definition file
# Bus 1 (pins 5, 3) is board SCL, SDA in the jetson definition file
# Default is to Bus 1; We are using Bus 0, so we need to construct the busio first ...
#print("Initializing Servos")
i2c_bus0=(busio.I2C(board.SCL_1, board.SDA_1))
#print("Initializing ServoKit")
kit = ServoKit(channels=16, i2c=i2c_bus0)
# kit[0] is the bottom servo
# kit[1] is the top servo
set_movement(in1_pin, in2_pin, 'f')
kit.servo[0].angle = int(18)
p.ChangeDutyCycle(0)
print("Done initializing")

# Disconnect
print("Aneu desconectant")
for j in range(45):
    print(45-j)
    time.sleep(1)


torch.cuda.set_device(0)

cam = Camera()
grid_params = {'xw':0.6, 'yw':+0.25, 'length_x':1.0, 'length_y':0.50, 'num_x':7,'num_y':7}
geompu = GeomPU(cam.get_K(), cam.get_Rt(),grid_params)

try:

    iterations = 0
    while(iterations<100):
        image = cam.get_frame_cv2()
        #image_path = "/home/dlinano/Pictures/Terrinus/16.jpeg"
        #image = cv2.imread(image_path)
        #cv2.imwrite("/home/dlinano/Pictures/exp_1/{:03d}.jpeg".format(iterations), image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(image)
        ori_width, ori_height = img.size

        # image transform, to torch float tensor 3xHxW
        #img = imresize(img, (512,384)) # imresize is width,height
        img = imresize(img, (480,640))
        #img = imresize(img, (640,480))
        img = img_transform(img)
        img = torch.unsqueeze(img, 0)
        #print("img size = " + str(img.size()))

        with torch.no_grad():
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=False)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
            enc_weights = "/home/dlinano/maweights/ade20k-mobilenetv2dilated-c1_deepsup/encoder_epoch_20.pth"
            net_encoder.load_state_dict(torch.load(enc_weights, map_location='cuda:0'), strict=False)
    
            net_decoder = C1DeepSup(num_class=150,fc_dim=320,use_softmax=True)
            dec_weights = "/home/dlinano/maweights/ade20k-mobilenetv2dilated-c1_deepsup/decoder_epoch_20.pth"
            net_decoder.load_state_dict(torch.load(dec_weights, map_location='cuda:0'), strict=False)
            net_encoder = net_encoder.cuda(0)
            net_decoder = net_decoder.cuda(0)
            auxi = net_encoder(img.cuda(0),return_feature_maps=True)
            segSizee = (480,640)
            pred = net_decoder(auxi, segSize=segSizee)
            scores = torch.zeros(1, 150, segSizee[0], segSizee[1]).cuda(0) # 150 num class
            scores = scores + pred
            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
            pred = (pred==3)*pred
            pred = np.uint8(pred)
            #cv2.imwrite("/home/dlinano/Pictures/exp_1_pred/{:03d}.jpeg".format(iterations), pred)
            BEV_img = geompu.create_BEV_image(pred)
            throttle_map, trajectory_BEV, trajectory_img = geompu.evaluate_grid(BEV_img)
            throttle_weights = np.array([0.1,0.1,0.1,0.1,0.2,0.2,0.2])
            max_throt_val = 75
            final_throttle = (max_throt_val/grid_params['num_y'])*np.sum(throttle_map*throttle_weights)
            steering_weights = np.array([0.1,0.1,0.1,0.1,0.2,0.2,0.2])
            h,w = BEV_img.shape
            steering_ys = (trajectory_BEV[1,:]/w)*36.0
            final_steering = np.sum(steering_weights*steering_ys)
            if final_steering > 18:
                final_steering = final_steering*1.5
            elif final_steering < 18:
                final_steering = final_steering*0.5
            kit.servo[0].angle = int(final_steering)
            p.ChangeDutyCycle(final_throttle)
            iterations = iterations + 1
        
finally:
    p.stop()
    GPIO.output(in2_pin, GPIO.LOW)
    GPIO.cleanup()

