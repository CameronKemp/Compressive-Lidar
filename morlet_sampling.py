#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:31:42 2023

@author: cameronkemp
"""








import numpy as np
from scipy.signal import convolve2d, morlet

def apply_sampling(array, central_frequency, noise_level, sz):
    morlet_wavelet = np.outer(np.real(morlet(sz, w=central_frequency)),np.real( morlet(sz, w=central_frequency)))
    white_noise = np.random.randn(sz, sz)

    sampling_functions = convolve2d(np.real(morlet_wavelet), white_noise, mode='same')
    sampling_functions = sampling_functions.reshape(sz, sz)

    obj_vector = np.reshape(array, (1, sz ** 2))

    m = np.zeros((sz ** 2, 1))
    for i in range(0, sz ** 2):
        m[i] = np.sum(sampling_functions[i, :] * obj_vector)

    # Assuming you want to add noise to the sampled image
    sampled_image = array + sampling_functions * noise_level

    return m, sampled_image

'''
def single_pixel(matrix, sz, array):
    if isinstance(matrix, tuple):
        matrix_array = tuple(np.array(m) for m in matrix)
    else:
        matrix_array = np.array(matrix)

    obj_vector = np.reshape(array, (1, sz ** 2))
    print(obj_vector.shape)

    m = np.zeros((sz ** 2, 1))
    for i in range(sz ** 2):
        m[i] = np.sum(matrix_array[i, :] * obj_vector)

    return m


'''
