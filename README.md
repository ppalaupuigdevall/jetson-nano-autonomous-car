# jetson-nano-autonomous-car
## Results v0.0
These results were obtained with the model MobileNetDilatedv2 - C1DeepSup
![territest_22_mobilenetdilatedv2-c1deepsup_2_5secs](https://user-images.githubusercontent.com/29488113/71564560-9b0c0900-2a57-11ea-9d65-6714fe532121.jpeg)

![GitHub Logo](/imgs/grid.jpeg)

Bird Eye View Image given Extrinsics and Intrinsics, defined by four points in WCS. 

![equation](https://latex.codecogs.com/gif.latex?K%20%3D%20%5Cbegin%7Bpmatrix%7D%20676.74%20%26%200%20%26%20317.4%5C%5C%200%20%26%20863.29%20%26%20252.459%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bpmatrix%7D)

Rotation matrix (World Coordinate System to Camera Coordinate System) WCS -> CCS, given by extrinsic rotation of the axis with angles X = 90, Y = 90, Z = 0.
T indicates WCS expressed in CCS
![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bpmatrix%7D%200.0%5C%5C%20-0.17%5C%5C%200.0%20%5Cend%7Bpmatrix%7D)


![GitHub Logo](/imgs/grid_BEV.jpeg)

#### Trajectory Estimation

![GitHub Logo](/imgs/grid_BEV_trajectory.jpeg)

![GitHub Logo](/imgs/grid_img_trajectory.jpeg)

## Setup
Enable Pin 33 / PWM2
´´´busybox devmem 0x70003248 32 0x46
busybox devmem 0x6000d100 32 0x00´´´


## Enllaços per fer memòria 
(Pull-up res)
https://www.arduino.cc/reference/en/language/variables/constants/constants/#:~:text=When%20a%20pin%20is%20configured,the%20pin%20(3.3V%20boards)
DC motor PWM
https://forums.developer.nvidia.com/t/what-pins-can-be-used-for-pwm-control-except-33-in-samples/78076/25
