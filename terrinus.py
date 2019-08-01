import numpy as np



class Terrinus:
    
    
    def __init__(self, camera, processing_unit, mapper, controller):
        self.camera = camera
        self.processing_unit = processing_unit
        self.mapper = mapper
        self.controller = controller
    
    
    def detect(self):
        mask = self.processing_unit.segment()

        return mask
    
    
    def create_map(self):
        map = None
        return map
    
    
    def compute_trajectory(self):
        trajectory = None
        return trajectory
    

    def compute_control(self):
        return (power, steering)