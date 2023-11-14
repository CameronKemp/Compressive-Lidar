
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plyfile import PlyData
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

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

def gabor_transform_analysis(image, frequencies, angles):
    for i, frequency in enumerate(frequencies):
        for j, angle in enumerate(angles):
            # Create Gabor filter
            gabor_filter = cv2.getGaborKernel((11, 11), 4.0, angle, frequency, 0.5, 0, ktype=cv2.CV_32F)

            # Apply Gabor filter to the image
            response = convolve2d(image, gabor_filter, mode='same', boundary='symm')

            # Use Gaussian filter to smooth the response
            response = gaussian_filter(response, sigma=1)

            # Display Gabor response in a separate figure
            plt.figure(figsize=(6, 6))
            plt.imshow(response, cmap='viridis')
            plt.title(f'Gabor Response (Frequency {i+1}, Angle {j+1})')
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

    # Define Gabor filter parameters
    frequencies = [0.1, 0.5, 1.0]  # You can adjust these frequencies
    angles = [0, 45, 90, 135]  # You can adjust these angles

    # Perform Gabor Transform analysis
    for i, frequency in enumerate(frequencies):
        for j, angle in enumerate(angles):
            # Create Gabor filter
            gabor_filter = cv2.getGaborKernel((11, 11), 4.0, angle, frequency, 0.5, 0, ktype=cv2.CV_32F)

            # Apply Gabor filter to the image
            response = convolve2d(df_reduced.to_numpy(), gabor_filter, mode='same', boundary='symm')

            # Use Gaussian filter to smooth the response
            response = gaussian_filter(response, sigma=1)

            # Display Gabor response in a separate figure
            plt.figure(figsize=(6, 6))
            plt.imshow(response, cmap='viridis')
            plt.title(f'Gabor Response (Frequency {i+1}, Angle {j+1})')
            plt.show()

            # Calculate SSIM
            ssim_value = ssim(df_reduced.to_numpy(), response, data_range=response.max() - response.min())
            print(f'SSIM (Frequency {i+1}, Angle {j+1}): {ssim_value}')




'''
The Gabor filter is a linear filter used in image processing and computer vision for texture analysis, edge detection, and feature extraction. It's particularly effective for capturing both high and low-frequency information in an image, making it suitable for analyzing textures with varying scales and orientations.

The Gabor filter is defined by a complex sinusoidal function modulated by a Gaussian function. The equation for a 2D Gabor filter in spatial domain is given by:

\[ G(x, y; \lambda, \theta, \psi, \sigma, \gamma) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(2\pi \frac{x'}{\lambda} + \psi\right) \]

where:
- \( (x', y') \) are the coordinates in a rotated coordinate system (rotated by an angle \( \theta \)),
- \( \lambda \) is the wavelength of the sinusoidal factor of the filter,
- \( \theta \) is the orientation of the Gabor filter,
- \( \psi \) is the phase offset,
- \( \sigma \) is the standard deviation of the Gaussian envelope, and
- \( \gamma \) is the spatial aspect ratio (elongation of the filter).

The Gabor filter is applied to an image by convolving it with the image. The resulting filtered image highlights structures that match the characteristics of the Gabor filter. Here are some key points about how Gabor filters work:

1. **Frequency and Orientation Selectivity**: The parameter \( \lambda \) controls the frequency of the sinusoidal component, determining the number of oscillations within the filter. The parameter \( \theta \) controls the orientation of the filter, allowing it to respond to structures with specific orientations.

2. **Phase Sensitivity**: The parameter \( \psi \) introduces a phase offset, making the filter sensitive to different phases of the sinusoidal component. This sensitivity is useful for capturing phase-related information in texture patterns.

3. **Spatial Aspect Ratio**: The parameter \( \gamma \) controls the spatial aspect ratio of the filter, allowing it to be elongated in a particular direction. This is useful for capturing structures with different aspect ratios.

4. **Gaussian Envelope**: The Gaussian envelope, controlled by \( \sigma \), determines the extent of the filter in space. It provides localization and ensures that the filter response diminishes smoothly away from the center.

In practical terms, Gabor filters are often used as a set of filters with different frequencies, orientations, and other parameters. This set of filters can be applied to an image to capture various texture features at different scales and orientations.

The filtered response of the Gabor filter can be visualized as an image, highlighting the regions in the input image that match the characteristics of the filter. This makes Gabor filters valuable in tasks such as texture discrimination, edge detection, and facial recognition.
'''