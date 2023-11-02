#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 22:56:25 2023

@author: cameronkemp
"""

import numpy as np
import pygame
from pygame.locals import *
from plyfile import PlyData
from OpenGL.GL import *
from OpenGL.GLU import *
import matplotlib.cm as cm
from sys import exit

def visualize_ply_data(file_path, point_size=5):
    try:
        # Read the PLY file
        ply_data = PlyData.read(file_path)
        vertices = ply_data['vertex']

        # Extract vertex coordinates
        x = vertices['x']
        y = vertices['y']
        z = vertices['z']

        # Normalize Z-values to determine colours
        min_z = np.min(z)
        max_z = np.max(z)
        normalized_z = (z - min_z) / (max_z - min_z)

        # Use a colormap to map normalized_z to colours
        colors = cm.viridis(normalized_z)

        # Initialize Pygame
        pygame.init()

        # Set up Pygame display
        screen_width = 800
        screen_height = 600
        screen = pygame.display.set_mode((screen_width, screen_height), OPENGL | DOUBLEBUF)
        pygame.display.set_caption("3D Visualization")

        # Enable smooth shading and depth testing
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        # Set initial rotation angles, vertical position, and zoom
        rotation_x = 0
        rotation_y = 45
        vertical_position = 0
        zoom_factor = 1.0  # Initial zoom factor

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT or event.type == K_UP:

                    running = False
                    pygame.quit()

                elif event.type == KEYDOWN:
                    if event.key == K_UP:
                        rotation_x += 5  # Increase pitch angle
                    elif event.key == K_DOWN:
                        rotation_x -= 5  # Decrease pitch angle
                    elif event.key == K_LEFT:
                        rotation_y += 5  # Increase yaw angle
                    elif event.key == K_RIGHT:
                        rotation_y -= 5  # Decrease yaw angle
                    elif event.key == K_PAGEUP:
                        vertical_position += 1.0  # Move up
                    elif event.key == K_PAGEDOWN:
                        vertical_position -= 1.0  # Move down
                    elif event.key == K_PLUS:
                        zoom_factor -= 5  # Zoom in
                    elif event.key == K_MINUS:
                        zoom_factor += 5  # Zoom out

            # Limit zoom factor to prevent extreme zoom levels
            zoom_factor = max(0.1, min(2.0, zoom_factor))

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Calculate the center of the point cloud
            center_x = np.mean(x)
            center_y = np.mean(y)
            center_z = np.mean(z)

            # Calculate the maximum dimension of the point cloud
            max_dimension = max(np.max(x) - np.min(x), np.max(y) - np.min(y), np.max(z) - np.min(z))

            # Calculate the scaling factor based on zoom
            scale_factor = 1.5 / max_dimension * zoom_factor

            # Set initial viewing angles and position
            glLoadIdentity()
            glRotatef(rotation_x, 1, 0, 0)  # Rotate around X-axis (pitch)
            glRotatef(rotation_y, 0, 1, 0)  # Rotate around Y-axis (yaw)
            glTranslatef(0, vertical_position, 0)  # Move vertically
            glScalef(scale_factor, scale_factor, scale_factor)
            glTranslatef(-center_x, -center_y, -center_z)

            # Apply vertical scale transformation to flip the image upside down
            glPushMatrix()
            glScalef(1, -1, 1)  # Apply vertical scale transformation
            glTranslatef(0, 0, 0)

            # Enable blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Scatter plot with colors
            glBegin(GL_POINTS)
            for xi, yi, zi, color in zip(x, y, z, colors):
                glColor4f(*color)  # Use the color from the colormap
                glVertex3f(xi, yi, zi)
            glEnd()

            glPopMatrix()  # Restore the transformation matrix

            pygame.display.flip()

        pygame.quit()

    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    #ply_file_path = r'/Users/cameronkemp/Documents/university/physics_project/physics_project/royale_20231005_110508_0.ply'
    ply_file_path = r'/Users/cameronkemp/Documents/university/physics_project/physics_project/royale_20231005_110844_0.ply'


    visualize_ply_data(ply_file_path, point_size=10)
