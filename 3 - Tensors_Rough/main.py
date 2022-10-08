# importing for the testing purpose
import numpy as np

# TENSORS
"""
    Scalars (0 - D Tensors) :- A tensor that contains only one
    number is known as a scalar or a 0 - D tensor, we can see this
    using the ndim property of numpy
"""
x = np.array(12 , dtype = "int")
print(x.ndim)

"""
    Vectors (1 - D Tensors) :- An array of numbers is know as 
    vectors or 1 - D tensors, A one Dimensional tensor have only
    one axis and can have multiple elements in that axis
"""
y = np.array([10 , 20 , 30 , 40] , dtype = "int")
print(y.ndim)

"""
    Matrices (2 - D tensors) :- An array of vectors is known as
    Matrices or 2 - D tensors, A Matrix has 2 axis often known as 
    rows and columsn
"""
z = np.array(
    [
        [10 , 20 , 30] , 
        [40 , 50 , 60]
    ] , 
    dtype = "int"
)
print(z.ndim)