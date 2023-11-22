

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
'''
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

    chaos_parameters = [0.1,0.5, 1.0]

    plt.figure(figsize=(20, 4))

    ssim_values = []  # Initialize the list

    for i, chaos_parameter in enumerate(chaos_parameters):
        # Apply chaotic sampling to the image
        chaotic_sampled_img = chaotic_sampling(org_img, chaos_parameter)

        # Calculate SSIM for the chaotic sampled image
        ssim_score_chaotic = ssim(org_img, chaotic_sampled_img, data_range=org_img.max() - org_img.min())
        print(f"Chaos Parameter: {chaos_parameter}, SSIM (Chaotic Sampled): {ssim_score_chaotic}")

        # Plot the chaotic sampled image in a subplot
        plt.subplot(1, len(chaos_parameters) + 1, i + 1)
        plt.imshow(chaotic_sampled_img, cmap='viridis')
        plt.title(f'Chaos Parameter: {chaos_parameter}\nSSIM: {ssim_score_chaotic:.4f}')
        plt.colorbar()

        ssim_values.append(ssim_score_chaotic)  # Append SSIM value to the list

    # Plot SSIM vs noise in the same subplot
    plt.subplot(1, len(chaos_parameters) + 1, len(chaos_parameters) + 1)
    plt.plot(chaos_parameters, ssim_values, marker='o', label='SSIM vs Noise')
    plt.title('SSIM vs Noise')
    plt.xlabel('Noise Level')
    plt.ylabel('SSIM Score')
    plt.legend()

    plt.tight_layout()
    plt.show()
'''

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

    chaos_parameters = [0.1, 0.5, 1.0]

    # Plot the original image
    plt.figure(figsize=(20, 4))
    plt.subplot(1, len(chaos_parameters) + 2, 1)
    plt.imshow(org_img, cmap='viridis')
    plt.title('Original Image')
    plt.colorbar()

    ssim_values = []  # Initialize the list
    ssim_nor_recon = []  # Initialize the list

    for i, chaos_parameter in enumerate(chaos_parameters):
        # Apply chaotic sampling to the image
        chaotic_sampled_img = chaotic_sampling(org_img, chaos_parameter)

        # Calculate SSIM for the chaotic sampled image
        ssim_score_chaotic = ssim(org_img, chaotic_sampled_img, data_range=org_img.max() - org_img.min())
        print(f"Chaos Parameter: {chaos_parameter}, SSIM (Chaotic Sampled): {ssim_score_chaotic}")

        # Plot the chaotic sampled image and SSIM in a subplot
        plt.subplot(1, len(chaos_parameters) + 2, i + 2)
        plt.imshow(chaotic_sampled_img, cmap='viridis')
        plt.title(f'Chaos Parameter: {chaos_parameter}\nSSIM: {ssim_score_chaotic:.4f}')
        plt.colorbar()

        ssim_values.append(ssim_score_chaotic)  # Append SSIM value to the list

        # Apply chaotic sampling to the image
        chaotic_sampled_img = chaotic_sampling(org_img, chaos_parameter)

        # Calculate SSIM for the reconstructed image
        ssim_score_recon = ssim(org_img, chaotic_sampled_img, data_range=org_img.max() - org_img.min())
        ssim_nor_recon.append(ssim_score_recon)  # Append SSIM value to the list

    # Plot SSIM vs noise in the same subplot
    plt.subplot(1, len(chaos_parameters) + 2, len(chaos_parameters) + 2)
    noise_levels = [0.1, 0.5, 5]
    ssim_values_noise = []

    for noise_level in noise_levels:
        n = np.random.randn(desired_size, desired_size)
        m_with_noise = org_img + n * noise_level

        sampled_transed_img = chaotic_sampling(m_with_noise, noise_level)

        recon = np.reshape(m_with_noise, (desired_size, desired_size))
        ssim_score_n = ssim(org_img, recon, data_range=org_img.max() - org_img.min())
        # ssim_nor.append(ssim_n)

        # Calculate SSIM for the chaotic sampled image
        ssim_score_sampled = ssim(org_img, sampled_transed_img, data_range=org_img.max() - org_img.min())
        ssim_values_noise.append(ssim_score_sampled)

    plt.plot(noise_levels, ssim_values_noise, marker='o', label='SSIM (Chaotic Sampled) vs Noise')
    plt.plot(noise_levels, ssim_nor_recon, marker='o', label='SSIM Original vs Noise')
    plt.title('SSIM vs Noise')
    plt.xlabel('Noise Level')
    plt.ylabel('SSIM Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
