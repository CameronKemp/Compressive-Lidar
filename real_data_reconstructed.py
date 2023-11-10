#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Nov  8 10:15:27 2023

@author: guyhamilton
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import pandas as pd
from plyfile import PlyData
from scipy.linalg import hadamard


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

def visualize_depth_image(depth_image, title):
    fig, ax = plt.subplots()
    ax.imshow(depth_image, cmap='viridis')
    ax.set_title(title)
    plt.show()

if __name__ == "__main__":
    # Load a PLY file
    ply_file_path = '/Users/guyhamilton/Desktop/Project Spyder/data.ply'

    ply_data = PlyData.read(ply_file_path)
    vertices = ply_data['vertex']

    # Extract vertex coordinates
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']

    # Create a 2D depth image (you can replace these values with your desired parameters)
    x_range = (-10, 10)
    y_range = (-10, 10)
    grid_width = 512
    grid_height = 512
    average_depth_image, _ = create_2d_depth_image(x, y, z, x_range, y_range, grid_width, grid_height)

    # Display the depth image
    visualize_depth_image(average_depth_image, 'Depth Image')

    # Define the desired size for the central region
    desired_size = 64

    # Calculate the starting and ending indices for rows and columns
    start_row = (512 - desired_size) // 2
    end_row = start_row + desired_size
    start_col = (512 - desired_size) // 2
    end_col = start_col + desired_size

    # Select the central region
    df_reduced = pd.DataFrame(average_depth_image).iloc[start_row:end_row, start_col:end_col]

    # Display the reduced DataFrame
    plt.imshow(df_reduced, cmap='viridis')
    plt.title('DataFrame Plot')
    plt.colorbar()
    plt.show()

    depth_array = df_reduced.to_numpy()

    # Define the dimensions of the depth image
    height, width = depth_array.shape

    # Create a grid of (x, y) coordinates corresponding to the pixels in the depth image
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D surface plot using the (x, y) coordinates and the depth values
    ax.plot_surface(x_grid, y_grid, depth_array, cmap='viridis')  # You can choose a different colormap

    # Customize the plot labels and title
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Depth Label')
    ax.set_title('3D Depth Image Visualization')

    # Show the 3D plot
    plt.show()

    # Calculate the histogram of depth values
    hist, depths = np.histogram(depth_array.ravel(), bins=10)
    plt.hist(depth_array.ravel(), bins=10)
    plt.title('Histogram of Depth versus intensity')
    plt.show()


H=hadamard(desired_size**2)
plt.imshow(H)
plt.show()

bins=10

difference_vals=[]
rec_vals=[]
sorted_weighted=[]
weighted_vals=[]

for i in range(0, desired_size**2, 1):
    H_new=(H[:,i])
    H_shaped=np.reshape(H_new, (desired_size, desired_size))
    
    
    H_positive = np.maximum(0, H_shaped)
    H_negative = abs(np.minimum(0, H_shaped))
    
    
    z_new_positive = depth_array * H_positive
    
    z_new_negative = depth_array * H_negative
    
    '''
    plotting new 3d plots and obtaining histograms of depth versus intensity
    '''

    hist_positive, depths_positive = np.histogram(z_new_positive, bins=bins)

    
    
    hist_negative, depths_negative = np.histogram(z_new_negative, bins=bins)

    
    z_difference=(hist_positive-hist_negative)
    difference_vals.append(z_difference)
    
    for i in z_difference:
        weighted=H_shaped*i
        weighted_vals.append(weighted)
        rec_vals.append(weighted)
        
'''
loop that splits up rec_vals into their original sets
'''
        
start = 0
end = len(rec_vals) 
step = bins
for i in range(start, end, step): 
    x = i 
    sorted_diff=(rec_vals[x:x+step])
    sorted_weighted.append(sorted_diff)

'''
loops over sorted_weighted to split them into their ith values and then sums them together
'''    

ith_index_start = 0  # Starting index
ith_index_end = bins  # Ending index (exclusive)
step = 1  # Step size for the loop


sums = []
# Iterate through the indices
for ith_index in range(ith_index_start, ith_index_end, step):
    # Sum of ith indices from each sub-array
    ith_sum = sum(array[ith_index] for array in sorted_weighted if ith_index < len(array))
    sums.append(ith_sum)

'''
for i in range(0, len(sorted_diff), 1):
    
    plt.imshow(sums[i])
    plt.title(f'image of sums at depth index {i}')
    plt.colorbar()
    plt.show()
'''

flattened_sums = [np.ravel(array) for array in sums]


max_arrays = []

# Iterate through each position in the flattened_sums arrays
for i in range(len(flattened_sums[0])):
    values_at_position = [array[i] for array in flattened_sums]
    max_index = np.argmax(values_at_position)
    max_arrays.append(max_index)
    

max_arrays_shaped=np.reshape(max_arrays, (desired_size, desired_size))

# Create an array to store the corresponding values from depths_positive
corresponding_depths = []

# Iterate through each index in max_arrays and use it to access values in depths_positive
for max_array_index in max_arrays:
    # Ensure max_array_index is within the valid range
    if 0 <= max_array_index < len(depths_positive):
        # Append the corresponding value from depths_positive to the corresponding_depths array
        corresponding_depths.append(depths_positive[max_array_index])
    else:
        # Handle the case where the index is out of range
        print(f"Invalid index: {max_array_index}")

# corresponding_depths now contains the corresponding values from depths_positive for each index in max_arrays



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_grid, y_grid, corresponding_depths, linewidths=0.5, alpha=0.7)
plt.show()



   