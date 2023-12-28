#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:53:36 2023

@author: cameronkemp
"""

from plyfile import PlyData

def load_ply_file(file_path):
    ply_data = PlyData.read(file_path)
    vertices = ply_data['vertex']
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    return x, y, z
