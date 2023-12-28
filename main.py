#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:47:45 2023

@author: cameronkemp
"""

import json
import matplotlib.pyplot as plt
from plyfile import PlyData
import numpy as np
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd
from depth_utils import *
from ply_loader import load_ply_file
from hadamard_transform import *
from combine_matrix import *
import matplotlib.pyplot as plt
from plyfile import PlyData
import numpy as np
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim

from scipy.linalg import hadamard
from skimage.metrics import structural_similarity as ssim
from measurement_analysis import *
from dct_matrix import *
from morlet_analysis import *
from fft_func import *

# Load configuration from JSON file
config_file_path = '/Users/cameronkemp/Documents/university/physics_project/refined_code_gray/refined_code/config.json'

with open(config_file_path, 'r') as f:
    config = json.load(f)

# Extract parameters from the configuration
ply_file_path = config["ply_file_path"]
depth_image_params = config["depth_image"]
central_region_params = config["central_region"]

# Read PLY file
ply_data = PlyData.read(ply_file_path)
vertices = ply_data['vertex']

# Extract vertex coordinates
x = vertices['x']
y = vertices['y']
z = vertices['z']

# Create a 2D depth image
x_range = depth_image_params["x_range"]
y_range = depth_image_params["y_range"]
grid_width = depth_image_params["grid_width"]
grid_height = depth_image_params["grid_height"]


# Define the desired size for the central region
sz = central_region_params["sz"]

noise_levels = config["noise_levels"]["noise_levels"]
#noise_levels = [10,50,100,1000]
array, img_vec,df_reduced = reduced_depth(x,y,z,x_range,y_range,grid_width,grid_height,sz)


###############################################################################


#/Users/cameronkemp/Documents/university/physics_project/physics_project
#obj,gray_img,img_vec = gray_scale(r'/Users/cameronkemp/Documents/university/physics_project/physics_project/Teddy256.png', sz)





obj = array

gray_img = df_reduced









#No Transform Applied

#no_transform(sz,noise_levels,gray_img,obj)



#Hadamard Transform

#matrix_transform = hadamard_matrix(img_vec, sz, array)

#combined = single_pixel(matrix_transform, sz, array)


#analysis(noise_levels,sz,matrix_transform,combined,gray_img,array)






###############################################################################
#DCT Transform

#matrix_transform,inverse_had,transed_img = dct_matrix(img_vec, sz, array)

#combined_dct = single_pixel(matrix_transform, sz, array)


#analysis(noise_levels,sz,matrix_transform,combined_dct,df_reduced,array)

###############################################################################

#FFT

#fft_analysis(array, sz, noise_levels)



###############################################################################

#Morlet Transform

morlet_analysis(df_reduced, sz,array,obj)

###############################################################################
