
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plyfile import PlyData
from skimage.metrics import structural_similarity as ssim
from skimage.transform import radon, iradon

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

def radon_transform_analysis(image, angles):
    # Perform Radon Transform
    radon_transform = radon(image, theta=angles, circle=True)

    # Perform Inverse Radon Transform (Backprojection)
    inverse_radon_transform = iradon(radon_transform, theta=angles, circle=True)

    # Display the original image, Radon Transform, and the reconstructed image
    plt.subplot(131), plt.imshow(df_reduced, cmap='viridis')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(radon_transform, cmap='viridis', extent=(0, 180, 0, radon_transform.shape[0]))
    plt.title('Radon Transform'), plt.xlabel('Projection Angle (degrees)'), plt.ylabel('Radon Transform')

    plt.subplot(133), plt.imshow(inverse_radon_transform, cmap='viridis')
    plt.title(f'Reconstructed Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def calculate_ssim(original, noisy):
    return ssim(original, noisy, data_range=noisy.max() - noisy.min())

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

    # Display the depth image
    visualize_depth_image(average_depth_image, 'Depth Image')

    # Define the desired size for the central region
    desired_size = 64
    start_row = (512 - desired_size) // 2
    end_row = start_row + desired_size
    start_col = (512 - desired_size) // 2
    end_col = start_col + desired_size
    # Select the central region
    df_reduced = df.iloc[start_row:end_row, start_col:end_col]

    # Define angles for Radon Transform
    angles = np.linspace(0, 180, 180, endpoint=False)

    # Define a range of noise levels
    noise_levels = [0, 0.1, 0.5, 1, 2, 4, 5, 10, 15, 20]  # Adjust the range and number of levels as needed
    ssims_value = []
    ssims_nor = []

    # Loop through the noise levels
    for noise_level in noise_levels:
        # Add noise to the reduced depth image
        noisy_depth_image = df_reduced.to_numpy() + np.random.normal(0, noise_level, df_reduced.shape)

        # Perform Radon Transform analysis for the noisy image
        radon_transform_analysis(noisy_depth_image, angles)

        # Perform SSIM analysis
        ssim_value = calculate_ssim(df_reduced.to_numpy(), noisy_depth_image)
        print(f"Noise Level: {noise_level}, SSIM (Radon): {ssim_value}")
        ssims_value.append(ssim_value)

        n = np.random.randn(desired_size, desired_size) * noise_level
        df_noisy = df_reduced + n

        ssim_nor = calculate_ssim(df_reduced.to_numpy(), df_noisy.to_numpy())
        print(f"Noise Level: {noise_level}, SSIM (Untransformed): {ssim_nor}")
        ssims_nor.append(ssim_nor)

    plt.plot(noise_levels, ssims_value, label='Radon')
    plt.plot(noise_levels, ssims_nor, label='Untransformed')
    plt.title('Radon vs Untransformed SSIM')
    plt.xlabel('Noise')
    plt.ylabel('SSIM')
    plt.legend()
    plt.show()