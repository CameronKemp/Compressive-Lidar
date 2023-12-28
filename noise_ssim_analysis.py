#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 22:14:37 2023

@author: cameronkemp
"""



import numpy as np
from scipy.linalg import hadamard
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd

fig, axs = plt.subplots(1, 5, figsize=(20, 4))



def analyze_measurements(data_array, sz,m,had_mat,noise_levels,df_reduced):

        ssims_plain = []
        ssims_had = []


       # Create a single row with 5 subplots
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))

        for i, noise_level in enumerate(noise_levels):

           n = np.random.randn(sz ** 2, 1)
           m_with_noise = m + n * noise_level

       # Reconstruct the image with noise
        recon = np.matmul(had_mat, m_with_noise)
        recon = np.reshape(recon, (sz, sz)) / (sz ** 2)
        n = np.random.randn(sz, sz) * noise_level

        df_noisy = df_reduced + n

        ssim_score_normal = ssim(data_array, df_noisy, data_range=data_array.max() - data_array.min())

       # Calculate SSIM
        ssim_score_had = ssim(data_array, recon, data_range=data_array.max() - data_array.min())
        print(f"Noise Level: {noise_level}, SSIM: {ssim_score_had}")
        ssims_plain += [ssim_score_normal]
        ssims_had += [ssim_score_had]

       # Display the reconstructed image for each noise level
        axs[i].imshow(recon, cmap='viridis')
        axs[i].set_title(f'Noise Level: {noise_level}, SSIM: {ssim_score_had}')

       # Add an additional subplot for SSIM values
        axs[-1].plot(noise_levels, ssims_plain, label='Untransformed')
        axs[-1].plot(noise_levels, ssims_had, label='Hadamard')
        axs[-1].set_title('SSIM Values')
        axs[-1].set_xlabel('Noise')
        axs[-1].set_ylabel('SSIM')
        axs[-1].legend()

        # Display the overall plot with subplots
        plt.tight_layout()
        plt.show()
