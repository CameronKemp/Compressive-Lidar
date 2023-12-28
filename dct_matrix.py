#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:44:25 2023

@author: cameronkemp
"""




import numpy as np
from scipy.fftpack import dct
import cv2

def dct_matrix(img_vec, sz, array):

    matrix_transform = dct(np.eye(img_vec.shape[0]), axis=0, norm='ortho')
    transed_img_vec = np.matmul(matrix_transform, img_vec)

    inverse_dct = np.matmul(matrix_transform.T, transed_img_vec)

    transed_img = inverse_dct.reshape(array.shape) / (sz ** 2)

    return matrix_transform, inverse_dct, transed_img
