#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:59:07 2023

@author: cameronkemp
"""


import matplotlib.pyplot as plt
from plyfile import PlyData
import numpy as np
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d, morlet

def load_ply(file_path):
    """
    Load PLY file and return vertex data.
    """
    return PlyData.read(file_path)['vertex']

def create_2d_depth_image(x, y, z, x_range, y_range, grid_width, grid_height):
    """
    Create a 2D depth image from 3D vertex data.
    """
    grid_cell_size_x = (x_range[1] - x_range[0]) / grid_width
    grid_cell_size_y = (y_range[1] - y_range[0]) / grid_height

    actual_grid_width = int((x_range[1] - x_range[0]) / grid_cell_size_x)
    actual_grid_height = int((y_range[1] - y_range[0]) / grid_cell_size_y)

    grid_z_sum = np.zeros((actual_grid_height, actual_grid_width), dtype=np.float32)
    grid_point_count = np.zeros((actual_grid_height, actual_grid_width), dtype=np.int32)

    for xi, yi, zi in zip(x, y, z):
        if x_range[0] <= xi <= x_range[1] and y_range[0] <= yi <= y_range[1]:
            grid_x = int((xi - x_range[0]) / grid_cell_size_x)
            grid_y = int((yi - y_range[0]) / grid_cell_size_y)
            grid_z_sum[grid_y, grid_x] += zi
            grid_point_count[grid_y, grid_x] += 1

    average_z_values = np.divide(grid_z_sum, grid_point_count, where=grid_point_count != 0)
    depth_image = cv2.resize(average_z_values, (grid_width, grid_height), interpolation=cv2.INTER_LINEAR)

    return average_z_values, depth_image

def visualize_depth_image(depth_image, title, cmap='viridis'):
    """
    Visualize the depth image.
    """
    fig, ax = plt.subplots()
    ax.imshow(depth_image, cmap=cmap)
    ax.set_title(title)
    plt.show()

def apply_sampling(image, central_frequency, noise_level):
    sz = image.shape[0]

    morlet_wavelet = np.outer(morlet(sz, w=central_frequency), morlet(sz, w=central_frequency))
    # Generate white Gaussian noise
    white_noise = np.random.randn(sz, sz)

    # Modified sampling functions - convolution of Morlet wavelet with white Gaussian noise
    sampling_functions = np.abs(convolve2d(morlet_wavelet, white_noise, mode='same'))

    sampling_functions = sampling_functions.reshape(sz, sz)

    # Apply sampling to the image
    sampled_image = image + sampling_functions * noise_level

    return sampled_image

def main():
    ply_file_path = '/Users/cameronkemp/Documents/university/physics_project/physics_project/royale_20231005_110844_0.ply'
    vertices = load_ply(ply_file_path)
    x, y, z = vertices['x'], vertices['y'], vertices['z']

    x_range = (-10, 10)
    y_range = (-10, 10)
    grid_width = 512
    grid_height = 512
    average_depth_image, _ = create_2d_depth_image(x, y, z, x_range, y_range, grid_width, grid_height)

    visualize_depth_image(average_depth_image, 'Depth Image')

    desired_size = 64

    start_row = (grid_height - desired_size) // 2
    end_row = start_row + desired_size
    start_col = (grid_width - desired_size) // 2
    end_col = start_col + desired_size

    df_reduced = pd.DataFrame(average_depth_image).iloc[start_row:end_row, start_col:end_col]

    org_img = df_reduced.to_numpy()

    plt.imshow(df_reduced, cmap='viridis')
    plt.title('DataFrame Plot')
    plt.colorbar()
    plt.show()

    img_vec = df_reduced.to_numpy().flatten()

    central_frequencies = [0.1, 0.2, 0.5, 1.0] #1.25, 1.5, 1.75, 2.0]  # Add more frequencies as needed
    noise_levels = [0,0.1,0.2,0.5,1,2,5,10,20]

    for central_frequency in central_frequencies:
        for noise_level in noise_levels:
            n = np.random.randn(desired_size, desired_size)
            m_with_noise = img_vec.reshape(desired_size, desired_size) + n * noise_level

            recon = np.reshape(m_with_noise, (desired_size, desired_size))

            # Calculate SSIM
            ssim_score = ssim(org_img, recon, data_range=org_img.max() - org_img.min())
            print(f"Noise Level: {noise_level}, Frequency: {central_frequency}, SSIM: {ssim_score}")

            df_recon = pd.DataFrame(recon)

            plt.imshow(recon, cmap='viridis')
            plt.colorbar()
            plt.title(f'Noise Level: {noise_level}, Frequency: {central_frequency}, SSIM: {ssim_score}')
            plt.show()

            # Apply the proposed sampling scheme to the transformed image
            sampled_transed_img = apply_sampling(recon, central_frequency, noise_level)

            # Calculate SSIM for the sampled image
            ssim_score_sampled = ssim(org_img, sampled_transed_img, data_range=org_img.max() - org_img.min())
            print(f"Noise Level: {noise_level}, Frequency: {central_frequency}, SSIM (Sampled): {ssim_score_sampled}")

            df_sampled_transed_img = pd.DataFrame(sampled_transed_img)

            plt.imshow(sampled_transed_img, cmap='viridis')
            plt.colorbar()
            plt.title(f'Sampled Transformed Image (Noise Level: {noise_level}, Frequency: {central_frequency}, SSIM: {ssim_score_sampled})')
            plt.show()

if __name__ == "__main__":
    main()
