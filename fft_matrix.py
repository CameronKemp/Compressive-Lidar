#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:50:50 2023

@author: cameronkemp
"""
import numpy as np
import cv2



def fft_transform_matrix(img_vec, sz, array):
    fft_result = np.fft.fft(img_vec)
    fft_result = np.abs(fft_result)
    fft_shifted = np.fft.fftshift(fft_result)
    return fft_shifted



import numpy as np

def fft_matrix(sz):

    rows = sz*sz
    cols = sz*sz
    fft_matrix = np.zeros((rows * cols, rows * cols), dtype=np.complex128)

    for i in range(rows):
        for j in range(cols):
            for k in range(rows):
                for l in range(cols):
                    angle = 2 * np.pi * ((i * k) / rows + (j * l) / cols)
                    fft_matrix[i * cols + j, k * cols + l] = np.exp(-1j * angle)

    return fft_matrix / np.sqrt(rows * cols)
