#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:50:30 2023

@author: cameronkemp
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from skimage.metrics import structural_similarity as ssim
import numpy as np
from scipy.signal import convolve2d, morlet
import cv2
import pandas as pd





import matplotlib.pyplot as plt
from plyfile import PlyData
import numpy as np
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d

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

def chaotic_sampling(image, chaos_parameter):
    """
    Apply chaotic sampling transformation to the image.
    """
    sz = image.shape[0]

    # Generate chaotic sequence
    chaotic_sequence = np.zeros((sz, sz))
    chaotic_sequence[0, 0] = 0.1  # Initial value

    for i in range(1, sz):
        chaotic_sequence[i, :] = chaotic_sequence[i - 1, :] + chaos_parameter * (1 - 2 * chaotic_sequence[i - 1, :])

    # Apply chaotic sampling to the image
    chaotic_sampled_image = image * chaotic_sequence

    return chaotic_sampled_image


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

    #img_vec = df_reduced.to_numpy().flatten()


    noise_levels = [5, 10, 50]
    chaos_parameter = 1e-30

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    depth_image = org_img

    sz = 64

    # Plot the original image
    axes[0].imshow(depth_image, cmap='viridis')
    axes[0].set_title('Noise Level: 0,SSIM: 1.0')
    axes[0].axis('off')

    ssim_values_cha = []
    ssim_nor_recon = []

    for i, noise_level in enumerate(noise_levels):
        n = np.random.randn(sz, sz)
        m_with_noise = depth_image + n * noise_level
        chaotic_sampled_img = chaotic_sampling(m_with_noise, chaos_parameter)

        ssim_score_chaotic = ssim(chaotic_sampled_img,depth_image, data_range=depth_image.max() - depth_image.min())

        # Plot the chaotic sampled image and SSIM in a subplot
        axes[i + 1].imshow(chaotic_sampled_img, cmap='viridis')
        axes[i + 1].set_title(f'Noise: {noise_level},SSIM: {ssim_score_chaotic:.4f}')
        axes[i + 1].axis('off')

        ssim_values_cha.append(ssim_score_chaotic)

        recon = np.reshape(m_with_noise, (sz, sz))
        ssim_nor_recon2 = ssim(recon,depth_image, data_range=depth_image.max() - depth_image.min())
        ssim_nor_recon.append(ssim_nor_recon2)

    # Plot SSIM vs Noise in the last subplot
    axes[-1].plot(noise_levels, ssim_nor_recon, marker='o', label='No Sample')
    axes[-1].plot(noise_levels, ssim_values_cha, marker='o', label='Chaotic Sampled')
    axes[-1].set_title('SSIM vs Noise')
    axes[-1].set_xlabel('Noise Level')
    axes[-1].set_ylabel('SSIM Score')
    axes[-1].legend()

    plt.suptitle(f'Chaotic Pattern Set: Chaos Parametre {chaos_parameter}', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
