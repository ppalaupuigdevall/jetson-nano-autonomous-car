import torch
import numpy as np
import cv2
import time
from torchvision import transforms
from PIL import Image
from dataset import imresize
from models.models import MobileNetV2Dilated
from models.models import C1DeepSup
import models.mobilenet as mobilenet
from terri_utils import img_transform
from terri_utils import round2nearest_multiple
# For DC motor
from adafruit_servokit import ServoKit # seteja el pinout a 1000 (GPIO.TEGRA_SOC)
import RPi.GPIO as GPIO

# For csi camera
from jetcam.csi_camera import CSICamera
# For Servo motor
import board
import busio

print("Aneu desconectant")
for j in range(40):
    print(40-j)
    time.sleep(1)

torch.cuda.set_device(0)
camera = CSICamera(width=640, height=480)

#print("Camera initialized")
image = camera.read()
cv2.imwrite('/home/dlinano/Pictures/pic1.jpeg', image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = Image.fromarray(image)
ori_width, ori_height = img.size

# image transform, to torch float tensor 3xHxW
img = img_transform(img)
img = torch.unsqueeze(img, 0)
#print("img size = " + str(img.size()))

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
    for i in range(0,2):
        auxi = net_encoder(img.cuda(0))
        pred = net_decoder(auxi, segSize=(480, 640))


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
#print("Done initializing")

steering = [18,0,36]
try:
    val = 90
    for i in range(3):
        kit.servo[0].angle = steering[i]
        p.ChangeDutyCycle(val)
        time.sleep(0.8)
        set_movement(in1_pin, in2_pin, 'b')
        p.ChangeDutyCycle(val)
        time.sleep(0.8)
        set_movement(in1_pin, in2_pin, 'f')
        
finally:
    p.stop()
    GPIO.output(in2_pin, GPIO.LOW)
    GPIO.cleanup()


