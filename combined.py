#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:01:12 2023

@author: cameronkemp
"""
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
    ply_file_path = '/Users/cameronkemp/Documents/university/physics_project/physics_project/royale_20231005_110844_0.ply'

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






    # Creating a Hadamard matrix of the right size
    hadamard_matrix = hadamard(depth_array.shape[0])


    hadamard_matrix[hadamard_matrix < 0] = 0  # Sets all -1 values to zero

    z_new = depth_array * hadamard_matrix

    # Create a 3D plot of the transformed depth data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D surface plot using the (x, y) coordinates and the depth values
    ax.plot_surface(x_grid, y_grid, z_new, cmap='viridis')  # You can choose a different colormap
    print(y_grid.max())
    # Customize the plot labels and title
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Depth Label')
    ax.set_title('3D Depth Image Visualization')
    plt.show()

# Calculate the histogram of transformed depth values
    hist, depths = np.histogram(z_new.ravel(), bins=10)
    plt.hist(z_new.ravel(), bins=10)
    plt.title('Histogram of Depth versus intensity')

    plt.show()
