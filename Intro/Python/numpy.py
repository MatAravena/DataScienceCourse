import numpy as np
import pandas as pd

arr_1= np.array([5,10,20])
arr_2= np.array([1,2,4])
print(arr_1 + arr_2)
print(arr_1 - arr_2)
print(arr_1 * arr_2)
print(arr_1 / arr_2)


print(arr_1 + 3.5)
print(arr_1 - 3.5)
print(arr_1 * 3.5)
print(arr_1 / 3.5)

#similar to lists:
print('Select first element of array with brackets:', arr_1[0])
print('Append new element to array using np.append():', np.append(arr_1,22.22))
print('Delete last element of array using np.delete():', np.delete(arr_1,-1)) #You can use the "-" sign to start from the end

print() #add empty line for better reading

#similar to pandas:
print('Calculate mean of array using np.mean():', np.mean(arr_1))
print('Calculate the median using np.median()', np.median(arr_1))
print('Calculate the median using np.quantile()', np.quantile(arr_1,0.5))



df = pd.read_csv('wine-quality.csv')
df.head()


type(df.values)
#numpy.ndarray

# N Dimentions
df.ndim                     # == 2
df.values.ndim              # == 2
df.loc[:,'pH'].values.ndim  # == 1


# Remember
# numpy offers many features that you already know from using DataFrames.
# Access an element in an ndarray like a DataFrame but without .iloc - my_array[row_num, col_num].
# Imagine the dimensions of an array as axes on which the data is arranged.
# A vector has one dimension. A matrix has two dimensions.