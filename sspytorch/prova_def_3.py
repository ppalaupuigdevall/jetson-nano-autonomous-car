import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from terri_utils import img_transform
from terri_utils import round2nearest_multiple
# For DC motor
from adafruit_servokit import ServoKit # seteja el pinout a 1000 (GPIO.TEGRA_SOC)
import RPi.GPIO as GPIO

from processing_unit import GeomPU
# For Servo motor
import board
import busio

import jetson.inference
import jetson.utils
import math
import argparse
import sys

from segnet_utils import *

"""
sudo python3 prova_def_3.py --input-flip=rotate-180 --input-width=640 --input-height=512 --visualize mask --stats true

"""

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
    #time.sleep(1)

def eulerAnglesToRotationMatrix(theta):
    """
    https://math.stackexchange.com/questions/2796055/3d-coordinate-rotation-using-roll-pitch-yaw
    """
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                               
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R



grid_params = {'xw':0.6, 'yw':+0.25, 'length_x':1.0, 'length_y':0.50, 'num_x':7,'num_y':7}
K = np.array([[676.74,0,317.4],[0,863.29,252.459],[0,0,1]])
angles = np.array([90,90,0])*np.pi/180   
R = eulerAnglesToRotationMatrix(angles) # Rotation matrix  WCS->CCS
T = np.array([[0.0],[-0.170],[0.0]]) # WCS expressed in camera coordinate system , before 0.15
Rt = np.hstack((R,T))
geompu = GeomPU(K, Rt,grid_params)

parser = argparse.ArgumentParser(description="terrinus first scripts", formatter_class = argparse.RawTextHelpFormatter, epilog=jetson.utils.videoSource.Usage())
parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")
opt = parser.parse_known_args()[0]

camera = jetson.utils.videoSource('csi://0', argv=sys.argv)
net = jetson.inference.segNet('fcn-resnet18-sun-640x512')
buffers = segmentationBuffers(net,opt)

# Disconnect
print("Aneu desconectant")
for j in range(45):
    print(45-j)
    time.sleep(1)

try:

    iterations = 0
    while(iterations<700):
        image = camera.Capture()
        buffers.Alloc(image.shape, image.format)
        net.Process(image)
        net.Mask(buffers.mask, width=640, height=512)
        mask = buffers.mask
        mask_np = jetson.utils.cudaToNumpy(mask)
        pred = mask_np[:,:,1]==128 # from segnet-console2.md jetson inference
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
            final_steering = final_steering*1.2
        elif final_steering < 18:
            final_steering = final_steering*0.8
        kit.servo[0].angle = int(final_steering)
        p.ChangeDutyCycle(final_throttle)
        iterations = iterations + 1
        
finally:
    camera.Close()
    p.stop()
    GPIO.output(in2_pin, GPIO.LOW)
    GPIO.cleanup()

