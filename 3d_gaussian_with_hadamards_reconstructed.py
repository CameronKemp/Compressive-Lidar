#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:17:25 2023

@author: guyhamilton
"""


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import hadamard
from matplotlib import cm
import plotly.graph_objects as go
from skimage import io, color

'''
Creating the 3D object
'''

max_dimension=64 #determines sze of plot
bins=10

x = np.arange(-10,10,20/max_dimension)
y = np.arange(-10,10,20/max_dimension)
X,Y = np.meshgrid(x,y)
Z = np.exp(-0.05*X**2-0.05*Y**2)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                     linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

hist_original, depth_original = np.histogram(Z, bins=bins)
plt.stairs(abs(hist_original), depth_original)
plt.ylabel('I(d) (number of pixels)')
plt.xlabel('Depth')
plt.title('Depth histogram of original object')
plt.show()

H=hadamard(max_dimension**2)
plt.imshow(H)
plt.show()

difference_vals=[]
rec_vals=[]
sorted_weighted=[]
weighted_vals=[]

for i in range(0, max_dimension**2, 1):
    H_new=(H[:,i])
    H_shaped=np.reshape(H_new, (max_dimension, max_dimension))
    
    
    H_positive = np.maximum(0, H_shaped)
    H_negative = abs(np.minimum(0, H_shaped))
    
    
    z_new_positive = Z * H_positive
    
    z_new_negative = Z * H_negative
    
    '''
    plotting new 3d plots and obtaining histograms of depth versus intensity
    '''

    hist_positive, depths_positive = np.histogram(z_new_positive, bins=bins)

    
    
    hist_negative, depths_negative = np.histogram(z_new_negative, bins=bins)

    
    z_difference=(hist_positive-hist_negative)
    difference_vals.append(z_difference)
    
    for i in z_difference:
        weighted=H_shaped*i
        weighted_vals.append(weighted)
        rec_vals.append(weighted)
        
'''
loop that splits up rec_vals into their original sets
'''
        
start = 0
end = len(rec_vals) 
step = bins
for i in range(start, end, step): 
    x = i 
    sorted_diff=(rec_vals[x:x+step])
    sorted_weighted.append(sorted_diff)

'''
loops over sorted_weighted to split them into their ith values and then sums them together
'''    

ith_index_start = 0  # Starting index
ith_index_end = bins  # Ending index (exclusive)
step = 1  # Step size for the loop


sums = []
# Iterate through the indices
for ith_index in range(ith_index_start, ith_index_end, step):
    # Sum of ith indices from each sub-array
    ith_sum = sum(array[ith_index] for array in sorted_weighted if ith_index < len(array))
    sums.append(ith_sum)

'''
for i in range(0, len(sorted_diff), 1):
    
    plt.imshow(sums[i])
    plt.title(f'image of sums at depth index {i}')
    plt.colorbar()
    plt.show()
'''

flattened_sums = [np.ravel(array) for array in sums]


max_arrays = []

# Iterate through each position in the flattened_sums arrays
for i in range(len(flattened_sums[0])):
    values_at_position = [array[i] for array in flattened_sums]
    max_index = np.argmax(values_at_position)
    max_arrays.append(max_index)
    

max_arrays_shaped=np.reshape(max_arrays, (max_dimension, max_dimension))

# Create an array to store the corresponding values from depths_positive
corresponding_depths = []

# Iterate through each index in max_arrays and use it to access values in depths_positive
for max_array_index in max_arrays:
    # Ensure max_array_index is within the valid range
    if 0 <= max_array_index < len(depths_positive):
        # Append the corresponding value from depths_positive to the corresponding_depths array
        corresponding_depths.append(depths_positive[max_array_index])
    else:
        # Handle the case where the index is out of range
        print(f"Invalid index: {max_array_index}")

# corresponding_depths now contains the corresponding values from depths_positive for each index in max_arrays



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, corresponding_depths, linewidths=0.5, alpha=0.7)
plt.show()
