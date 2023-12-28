#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:10:35 2023

@author: cameronkemp
"""


import json
import matplotlib.pyplot as plt
from plyfile import PlyData
import numpy as np
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
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
from morlet_analysis import *


'''
def fft_analysis(array, sz, noise_levels):
    # Set the desired size for FFT analysis


    # Resize the image to the desired size
    original_image = array

    # Perform FFT
    fft_result = np.fft.fft2(original_image)
    fft_shifted = np.fft.fftshift(fft_result)
    # Perform Inverse FFT
    inverse_fft = np.fft.ifftshift(fft_shifted)
    reconstructed_image = np.abs(np.fft.ifft2(inverse_fft))

    # ADDITIONAL NOISE AND SSIM ANALYSIS
    ssims_fft = []
    ssims_nor = []
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))


    for j, noise_level in enumerate(noise_levels):

        n = np.random.randn(sz, sz)
        m_with_noise = fft_result + n * noise_level

        # Reconstruct the image with noise using FFT
        recon = np.abs(np.fft.ifft2(np.fft.ifftshift(m_with_noise)))

        ssim_score_fft = ssim_score_fft = round(ssim(np.real(original_image), np.real(recon), data_range=original_image.max() - original_image.min()), 4)


        recon3 = np.reshape(m_with_noise, (sz, sz))
        ssim_nor_recon2 = round(ssim(recon3,array, data_range=array.max() - array.min()),4)
        ssim_nor_recon.append(ssim_nor_recon2)



        print(f"Noise Level: {noise_level}\nSSIM (Sampled): {ssim_score_fft}")

        axs[j].imshow(recon, cmap='viridis')  # Fixing variable name here
        axs[j].set_title(f'Noise Level: {noise_level}, SSIM: {ssim_score_fft}')
        axs[j].axis('off')
        ssims_fft.append(ssim_score_sampled)

        ssims_fft.append(ssim_score_fft)  # Fixing variable name here
        ssim_nor.append(ssim_score_sampled)

        axs[1].set_xlabel('Noise Level')
        axs[0].axis('off')  # Blank subplot for better layout

     # Plot SSIM against noise in the final subplot
    axs[4].plot(noise_levels, ssim_values, marker='o', label = 'FFT Transform')
    axs[4].plot(noise_levels, ssim_nor, marker='x',label = 'Untransformed')
    axs[4].legend()
    axs[4].set_title('SSIM vs Noise')
    fig.suptitle('FFT Transform')
    axs[4].set_xlabel('Noise Level')

    plt.tight_layout()
    plt.show()
'''

def fft_analysis(array, sz, noise_levels):
    # Set the desired size for FFT analysis


    # Resize the image to the desired size
    original_image = array

    # Perform FFT
    fft_result = np.fft.fft2(original_image)
    fft_shifted = np.fft.fftshift(fft_result)
    # Perform Inverse FFT
    inverse_fft = np.fft.ifftshift(fft_shifted)
    reconstructed_image = np.abs(np.fft.ifft2(inverse_fft))

    # ADDITIONAL NOISE AND SSIM ANALYSIS
    ssims_fft = []
    ssims_nor = []
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))


    for i, noise_level in enumerate(noise_levels):
        n = np.random.randn(sz, sz)
        m_with_noise = fft_result + (n * noise_level)


        # Reconstruct the image with noise using FFT
        fft_sampled = np.abs(np.fft.ifft2(np.fft.ifftshift(m_with_noise)))

        fft_ssim = round(ssim(original_image, fft_sampled, data_range=original_image.max() - original_image.min()),4)



        n = np.random.randn(sz, sz) * noise_level
        df_noisy = original_image + n

        ssim_nor = ssim(original_image, df_noisy, data_range=original_image.max() - original_image.min())
        #print(f"Noise Level: {noise_level}, SSIM: {ssim_score_fft}")





        axs[i].imshow(fft_sampled, cmap='viridis')  # Fixing variable name here
        axs[i].set_title(f'Noise Level: {noise_level}, SSIM: {fft_ssim}')
        axs[i].axis('off')


        ssims_fft.append(fft_ssim)

        ssims_nor.append(ssim_nor)

        axs[1].set_xlabel('Noise Level')
        axs[0].axis('off')  # Blank subplot for better layout

     # Plot SSIM against noise in the final subplot

    axs[4].plot(noise_levels, ssims_nor, marker='x',label = 'Untransformed')
    axs[4].plot(noise_levels, ssims_fft, marker='o', label = 'FFT Transform')
    axs[4].legend()
    axs[4].set_title('SSIM vs Noise')
    fig.suptitle('FFT Transform', fontsize = 16)
    axs[4].set_xlabel('Noise Level')

    plt.tight_layout()
    plt.show()
