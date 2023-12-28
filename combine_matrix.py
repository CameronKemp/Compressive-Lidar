#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:12:31 2023

@author: cameronkemp
"""

import numpy as np



def single_pixel(matrix, sz, array):
    if isinstance(matrix, tuple):
        matrix_array = tuple(np.array(m) for m in matrix)
    else:
        matrix_array = np.array(matrix)

    obj_vector = np.reshape(array, (1, sz ** 2))

    m = np.zeros((sz ** 2, 1))

    for i in range(sz ** 2):
        m[i] = np.sum(matrix_array[i, :] * obj_vector)


    return m
