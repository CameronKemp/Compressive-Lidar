#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:28:19 2023

@author: cameronkemp
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plyfile import PlyData
from skimage.metrics import structural_similarity as ssim

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

def fft_analysis(image, desired_size, noise_levels):
    # Set the desired size for FFT analysis
    sz = desired_size

    # Resize the image to the desired size
    original_image = cv2.resize(image, (sz, sz))

    # Perform FFT
    fft_result = np.fft.fft2(original_image)
    fft_shifted = np.fft.fftshift(fft_result)

    # Calculate magnitude spectrum (logarithmic scale for better visualization)
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)

    # Perform Inverse FFT
    inverse_fft = np.fft.ifftshift(fft_shifted)
    reconstructed_image = np.abs(np.fft.ifft2(inverse_fft))

    # ADDITIONAL NOISE AND SSIM ANALYSIS
    ssims_fft = []
    ssims_nor = []
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))


    for i, noise_level in enumerate(noise_levels):
        n = np.random.randn(sz, sz)
        m_with_noise = fft_result + n * noise_level

        # Reconstruct the image with noise using FFT
        recon = np.fft.ifft2(np.fft.ifftshift(m_with_noise)).real

        ssim_score_fft = ssim(original_image, recon, data_range=original_image.max() - original_image.min())
        print(f"Noise Level: {noise_level}, SSIM: {ssim_score_fft}")
        ssims_fft.append(ssim_score_fft)


        n = np.random.randn(desired_size, desired_size) * noise_level
        df_noisy = original_image + n

        ssim_score_normal = ssim(original_image, df_noisy, data_range=original_image.max() - original_image.min())
        print(f"Noise Level: {noise_level}, SSIM: {ssim_score_fft}")
        ssims_nor.append(ssim_score_normal)

        axs[i].imshow(recon, cmap='viridis')
        #axs[i].set_title(f'Noise Level: {noise_level}, SSIM: {ssim_score_had}')

# Add an additional subplot for SSIM values
    axs[-1].plot(noise_levels, ssims_fft, label='Untransformed')
    axs[-1].plot(noise_levels, ssims_nor, label='Hadamard')
    axs[-1].set_title('SSIM Values')
    axs[-1].set_xlabel('Noise')
    axs[-1].set_ylabel('SSIM')
    axs[-1].legend()

# Display the overall plot with subplots
plt.tight_layout()
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

    # Create a 2D depth image
    x_range = (-10, 10)
    y_range = (-10, 10)
    grid_width = 512
    grid_height = 512
    average_depth_image, _ = create_2d_depth_image(x, y, z, x_range, y_range, grid_width, grid_height)

    # Convert the depth image to a DataFrame
    df = pd.DataFrame(average_depth_image)


    # Define the desired size for the central region
    desired_size = 64
    start_row = (512 - desired_size) // 2
    end_row = start_row + desired_size
    start_col = (512 - desired_size) // 2
    end_col = start_col + desired_size
    # Select the central region
    df_reduced = df.iloc[start_row:end_row, start_col:end_col]

    # Perform FFT analysis with additional noise and SSIM analysis
    fft_analysis(df_reduced.to_numpy(), desired_size, [0, 0.1, 5, 10])
