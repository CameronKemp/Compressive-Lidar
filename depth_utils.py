#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:53:35 2023

@author: cameronkemp
"""

import cv2
import numpy as np
import matplotlib as plt
import pandas as pd

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
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from skimage.metrics import structural_similarity as ssim
from measurement_analysis import *
from dct_matrix import *




def create_2d_depth_image(x, y, z, x_range, y_range, grid_width, grid_height):

   # Calculate the size of each grid cell
   grid_cell_size_x = (x_range[1] - x_range[0]) / grid_width
   grid_cell_size_y = (y_range[1] - y_range[0]) / grid_height

   # Determine the actual grid dimensions based on the available data
   actual_grid_width = int((x_range[1] - x_range[0]) / grid_cell_size_x)
   actual_grid_height = int((y_range[1] - y_range[0]) / grid_cell_size_y)

   # Create an array to store the sum of Z values and count of points in each grid cell
   grid_z_sum = np.zeros((actual_grid_height, actual_grid_width), dtype=np.float32)
   grid_point_count = np.zeros((actual_grid_height, actual_grid_width), dtype=np.int32)

   # Iterate through your 3D data and calculate average Z values
   for xi, yi, zi in zip(x, y, z):
       if x_range[0] <= xi <= x_range[1] and y_range[0] <= yi <= y_range[1]:
           # Calculate the grid cell indices for the point
           grid_x = int((xi - x_range[0]) / grid_cell_size_x)
           grid_y = int((yi - y_range[0]) / grid_cell_size_y)
           # Add Z value to the grid cell sum and increment the point count
           grid_z_sum[grid_y, grid_x] += zi
           grid_point_count[grid_y, grid_x] += 1

   # Calculate the average Z values for each grid cell (avoid division by zero)
   average_z_values = np.divide(grid_z_sum, grid_point_count, where=grid_point_count != 0)

   # Create a depth image with the actual grid dimensions
   depth_image = cv2.resize(average_z_values, (grid_width, grid_height), interpolation=cv2.INTER_LINEAR)


   return average_z_values, depth_image


def reduced_depth(x,y,z,x_range,y_range,grid_width,grid_height,sz):
    create_2d_depth_image(x, y, z, x_range, y_range, grid_width, grid_height)

    average_depth_image, _ = create_2d_depth_image(x, y, z, x_range, y_range, grid_width, grid_height)

    # Convert the depth image to a DataFrame
    df = pd.DataFrame(average_depth_image)

    start_row = (512 - sz) // 2
    end_row = start_row + sz
    start_col = (512 - sz) // 2
    end_col = start_col + sz

    # Select the central region
    df_reduced = df.iloc[start_row:end_row, start_col:end_col]

    array = df_reduced.to_numpy()


    img_vec = array.flatten()


    return array, img_vec,df_reduced



def visualize_depth_image(depth_image, title, cmap='viridis'):
    """
    Visualize the depth image.
    """
    fig, ax = plt.subplots()
    ax.imshow(depth_image, cmap=cmap)
    ax.set_title(title)
    plt.show()





def gray_scale(path,sz):
    raw_img = cv2.imread(path)
    plt.imshow(raw_img)
    plt.show()
    gray_img = cv2.resize(raw_img, (sz,sz))
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    # Convert to number array
    obj = np.array(gray_img)
    plt.title('Ground truth')
    plt.imshow(obj)
    img_vec = obj.flatten()

    return(obj,gray_img,img_vec)
