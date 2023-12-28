#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:38:34 2023

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
from morlet_samp import *




def morlet_analysis(df_reduced,sz,array,org):

    org_img = org
    img_vec = org.flatten()


    central_frequencies = [0.01]
    noise_levels = [1, 5, 10]

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    axs[0].imshow(org_img, cmap='viridis')
    axs[0].set_title('Noise Level: 0,SSIM: 1.0')

    ssim_values = []  # Collect SSIM values for each noise level
    ssim_nor = []

    obj_Vector = np.reshape(array, (1, sz ** 2))
    m = np.zeros((sz ** 2, 1))
    '''
    for i in range(0, sz ** 2):
        m[i] = np.sum(sampled_image[i, :] * obj_Vector)
    '''
    for j, noise_level in enumerate(noise_levels):



        n = np.random.randn(sz, sz)
        m_with_noise = img_vec.reshape(sz, sz) + n * noise_level
        #m_with_noise = m + n * noise_level
        recon = np.reshape(m_with_noise, (sz, sz))

        sampled_transed_img = apply_sampling_morlet(recon, central_frequencies[0], noise_level)

        ssim_score_sampled = round(ssim(org_img, sampled_transed_img, data_range=org_img.max() - org_img.min()),4)

        print(f"Noise Level: {noise_level}, SSIM (Sampled): {ssim_score_sampled}")

        axs[j + 1].imshow(sampled_transed_img, cmap='viridis')
        axs[j + 1].set_title(f'Noise Level: {noise_level}, SSIM: {ssim_score_sampled}')
        axs[j+1].axis('off')
        ssim_values.append(ssim_score_sampled)

        ssim_score= round(ssim(org_img,recon, data_range=org_img.max() - org_img.min()),4)

        ssim_nor.append(ssim_score)

    axs[1].set_xlabel('Noise Level')
    axs[0].axis('off')  # Blank subplot for better layout

    # Plot SSIM against noise in the final subplot
    axs[4].plot(noise_levels, ssim_nor, marker='x',label = 'Untransformed')
    axs[4].plot(noise_levels, ssim_values, marker='o', label = 'Sampled')
    axs[4].legend()
    axs[4].set_title('SSIM vs Noise')
    fig.suptitle(f'Morlet Transform Frequency: {central_frequencies[0]}', fontsize = 16)
    axs[4].set_xlabel('Noise Level')

    plt.tight_layout()
    plt.show()
