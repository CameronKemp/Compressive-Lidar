#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:33:24 2023

@author: cameronkemp
"""


import numpy as np
from scipy.signal import convolve2d, morlet


def apply_sampling_morlet(image, central_frequency, noise_level):
    sz = image.shape[0]

    morlet_wavelet = np.outer(morlet(sz, w=central_frequency), morlet(sz, w=central_frequency))
    white_noise = np.random.randn(sz, sz)

    sampling_functions = np.real(convolve2d(morlet_wavelet, white_noise, mode='same'))
    sampling_functions = sampling_functions.reshape(sz, sz)

    sampled_image = image + sampling_functions * noise_level

    return sampled_image
