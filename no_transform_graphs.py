#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:58:09 2023

@author: cameronkemp
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:44:04 2023

@author: cameronkemp
"""


import matplotlib.pyplot as plt
from plyfile import PlyData
import numpy as np
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from skimage.metrics import structural_similarity as ssim
from depth_utils import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



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

    # Convert the depth image to a DataFrame
    df = pd.DataFrame(average_depth_image)


# Define the desired size for the central region
desired_size = 64

# Calculate the starting and ending indices for rows and columns
start_row = (512 - desired_size) // 2
end_row = start_row + desired_size
start_col = (512 - desired_size) // 2
end_col = start_col + desired_size

# Select the central region
df_reduced = df.iloc[start_row:end_row, start_col:end_col]

array = df_reduced.to_numpy()

img_vec = array.flatten()
Had_mat = hadamard(img_vec.shape[0])
transed_img_vec = np.matmul(Had_mat, img_vec)

inverse_had = np.matmul(Had_mat, transed_img_vec)

transed_img = inverse_had.reshape(df_reduced.shape) / (desired_size ** 2)

sz = 64
obj = array

H_h = hadamard(sz**2)

# Measure for each pattern
obj_Vector = np.reshape(obj, (1, sz ** 2))
m = np.zeros((sz ** 2, 1))

for i in range(0, sz ** 2):
    m[i] = np.sum(H_h[i, :] * obj_Vector)

# (The rest of the code you provided for measurements and reconstructions)

# ADDITIONAL NOISE AND SSIM ANALYSIS
noise_levels = [0, 0.1, 0.5, 5]

ssims_plain = []
ssims_had = []

# Create a single row with 5 subplots
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i, noise_level in enumerate(noise_levels):
    n = np.random.randn(sz ** 2, 1)
    m_with_noise = m + n * noise_level

    n = np.random.randn(desired_size, desired_size) * noise_level
    df_noisy = df_reduced + n

    ssim_score_normal = ssim(array, df_noisy, data_range=array.max() - array.min())

    # Calculate SSIM
    #ssim_score_had = ssim(array, recon, data_range=array.max() - array.min())
    print(f"Noise Level: {noise_level}, SSIM: {ssim_score_normal}")
    ssims_plain += [ssim_score_normal]
    #ssims_had += [ssim_score_had]

    # Display the reconstructed image for each noise level
    axs[i].imshow(df_noisy, cmap='viridis')
    axs[i].set_title(f'Noise Level: {noise_level}, SSIM: {ssim_score_normal}')

# Add an additional subplot for SSIM values
axs[-1].plot(noise_levels, ssims_plain, label='Untransformed')
#axs[-1].plot(noise_levels, ssims_had, label='Hadamard')
axs[-1].set_title('SSIM Values - No Transformation')
axs[-1].set_xlabel('Noise')
axs[-1].set_ylabel('SSIM')
axs[-1].legend()

# Display the overall plot with subplots
plt.tight_layout()
plt.show()
