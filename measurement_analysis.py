#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:53:37 2023

@author: cameronkemp
"""

import numpy as np
from scipy.linalg import hadamard
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd
from combine_matrix import *


def analysis(noise_levels, sz, matrix, m, df_reduced, array):
    ssims_plain = []
    ssims_transformed = []
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    for i, noise_level in enumerate(noise_levels):
        n = np.random.randn(sz ** 2, 1)
        m_with_noise = m + n * noise_level

        # Reconstruct the image with noise
        recon = np.matmul(matrix, m_with_noise)
        recon = np.reshape(recon, (sz, sz)) / (sz ** 2)
        n = np.random.randn(sz, sz) * noise_level
        df_noisy = df_reduced + n

        ssim_score_normal = round(ssim(array, df_noisy, data_range=array.max() - array.min()),4)

        # Calculate SSIM
        ssim_score_transformed = round(ssim(array, recon, data_range=array.max() - array.min()),4)

        ssims_plain.append(ssim_score_normal)
        ssims_transformed.append(ssim_score_transformed)

        # Display the reconstructed image for each noise level
        axs[i].imshow(recon, cmap='viridis')
        axs[i].set_title(f'Noise Level: {noise_level},SSIM: {ssim_score_transformed}')
        axs[i].axis('off')

    # Add an additional subplot for SSIM values
    axs[-1].plot(noise_levels, ssims_plain, label='Untransformed')
    axs[-1].plot(noise_levels, ssims_transformed, label='Transformed')
    axs[-1].set_title('SSIM Values')
    axs[-1].set_xlabel('Noise')

    plt.suptitle(f'Hadamard', fontsize=16)
    axs[-1].set_ylabel('SSIM')
    axs[-1].legend()

    # Display the overall plot with subplots
    plt.tight_layout()
    plt.show()






def no_transform(sz,noise_levels,df_reduced,array):


    ssims_plain = []

    # Create a single row with 5 subplots
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    for i, noise_level in enumerate(noise_levels):
        n = np.random.randn(sz ** 2, 1)
        m_with_noise = n * noise_level
        n = np.random.randn(sz, sz) * noise_level
        df_noisy = df_reduced + n

        ssim_score_normal = round(ssim(array, df_noisy, data_range=array.max() - array.min()),4)


        #print(f"Noise Level: {noise_level}, SSIM: {ssim_score_normal}")
        ssims_plain += [ssim_score_normal]

        axs[i].imshow(df_noisy, cmap='viridis')
        axs[i].set_title(f'Noise Level: {noise_level}, SSIM: {ssim_score_normal}')
        axs[i].axis('off')

    # Add an additional subplot for SSIM values
    axs[-1].plot(noise_levels, ssims_plain, label='Untransformed')
    #axs[-1].plot(noise_levels, ssims_had, label='Hadamard')
    axs[-1].set_title('SSIM Values')
    axs[-1].set_xlabel('Noise')
    axs[-1].set_ylabel('SSIM')
    fig.suptitle('No Transform')
    axs[-1].legend()

    # Display the overall plot with subplots
    plt.tight_layout()
    plt.show()
