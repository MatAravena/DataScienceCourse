import numpy as np
vec = np.array([1,2])
vec_2 = np.array([2,1])
vec_sum = vec + vec_2 
vec_sum

# Lengths of vectors: The norm
vec_length = 0
for i in range(len(vec)): # loop throug every value in vec
    vec_length = vec_length + (vec[i]**2) #sum up the squares of the values
vec_length = np.sqrt(vec_length) #take the square root with np.sqrt()
print(vec_length)

# Numpy Linear Algebra module
from numpy.linalg import norm
# To calculate the length of the vector
norm(vec)


# Euclidean distance
# is the distance between 2 points in a Matrix


# Dot Product   / Scalar product
# Calculate the angule from 2 vectors
angle_vec_vec2 = np.dot(vec, vec_2) / (norm(vec) * norm(vec_2))
angle_vec_vec2

# Strictly speaking, the angle measure is the cosine function of the angle. The cosine is a function that converts angles into numbers between -1 and +1.

# To get the angle now, you have to apply the function np.arccos() to angle_vec_vec2 and angle_vec_vec.
# The arccosine is the inverse function of the cosine. We use it to calculate an angle from a number between -1 and 1. 
# However, we get an angle that is not given in degrees as usual, but in radians, which are used more often in mathematics.
# To make the angle itself easier to read, we can convert it to an angle in degrees with the function np.degrees(). What angle do we get in each case? Print them both.

np.degrees(np.arccos(angle_vec_vec))
np.degrees(np.arccos(angle_vec_vec_2))


# Remember:

# Use the norm() function from numpy.linalg to calculate the length of vectors.
# Calculate the dot product of two vectors with np.dot()
# Calculate the cosine of the angle between two vectors using the ratio of the dot product and the product of the norm of the two vectors
