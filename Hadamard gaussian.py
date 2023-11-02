# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import hadamard
from matplotlib import cm
import plotly.graph_objects as go
from skimage import io, color


max_dimension=16 #determines sze of plot

x = np.arange(-10,10,20/max_dimension)
y = np.arange(-10,10,20/max_dimension)
X,Y = np.meshgrid(x,y)
Z = np.exp(-0.05*X**2-0.05*Y**2)

fig = go.Figure(data=[go.Surface(z=Z)])
fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor='limegreen', project_z=True))

fig.update_layout(autosize=True, width=500, height=500)

fig.show()

hist,depths=np.histogram(Z, bins=10)
plt.stairs(hist, depths)
plt.title('Histogram of Depth versus intensity')
    
print(hist)
print(depths)
plt.show()


def hadamard_recursive(n):
    if n == 1:
        return np.array([[1]])
    else:
        H_small = hadamard_recursive(n // 2)
        H = np.block([[H_small, H_small],
                      [H_small, -H_small]])
        return H

def scale_hadamard_matrix(matrix, target_dimension):
    current_dimension = matrix.shape[0]
    if current_dimension == target_dimension:
        return matrix
    else:
        # Calculate the scaling factor for the Kronecker product
        scale_factor = target_dimension // current_dimension
        # Use Kronecker product to scale the matrix
        scaled_matrix = np.kron(matrix, np.ones((scale_factor, scale_factor)))
        return scaled_matrix

def generate_hadamard_matrices(up_to_dimension):
    hadamard_matrices = {}
    for order in range(2, up_to_dimension + 1):
        if order & (order - 1) == 0:  # Check if order is a power of 2
            H = hadamard_recursive(order)
            scaled_H = scale_hadamard_matrix(H, up_to_dimension)
            hadamard_matrices[order] = scaled_H
    return hadamard_matrices

# Generate Hadamard matrices of each order up to max_dimension and scale them up
hadamard_matrices = generate_hadamard_matrices(max_dimension)

# Print the generated Hadamard matrices
for order, H in hadamard_matrices.items():
    print(f"Hadamard Matrix of Order {order} (scaled to {max_dimension}x{max_dimension}):\n{H}\n")
    
    
    # Generate Hadamard matrices of each order up to max_dimension and scale them up
hadamard_matrices = generate_hadamard_matrices(max_dimension)

I_positive=[]

# Print the generated Hadamard matrices
for order, H in hadamard_matrices.items():
    H[H < 0] = 0 #sets all -1 values to zero
    
    z_new=Z*H
    
    fig = go.Figure(data=[go.Surface(z=z_new)])
    fig.update_layout(autosize=True, width=500, height=500, title='Surface with projected Hadamard')
    fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor='limegreen', project_z=True))
    
    hist,depths=np.histogram(z_new, bins=10)
    plt.stairs(hist, depths)
    plt.title('Histogram of Depth versus intensity')
    
    fig.show()
    
    print(hist)
    print(depths)
    plt.show()
    
    # Print the generated Hadamard matrices
for order, H in hadamard_matrices.items():
    H[H > 0] = 0 #set all +1 values to zero
    
    z_new_negative=Z*H
    
    fig = go.Figure(data=[go.Surface(z=z_new_negative)])
    fig.update_layout(autosize=True, width=500, height=500, title='Surface with projected Hadamard')
    fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor='limegreen', project_z=True))
    
    hist,depths=np.histogram(z_new_negative, bins=10)
    plt.stairs(hist, depths)
    plt.title('Histogram of Depth versus intensity')
    
    fig.show()
    
    print(hist)
    print(depths)
    plt.show()