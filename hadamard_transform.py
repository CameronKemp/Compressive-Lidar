#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:53:36 2023

@author: cameronkemp
"""

import numpy as np
from scipy.linalg import hadamard
import cv2

def hadamard_matrix(img_vec, sz, array):
    matrix_transform = hadamard(img_vec.shape[0])
    transed_img_vec = np.matmul(matrix_transform, img_vec)



    return matrix_transform
