import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import solve

# Helper Functions

def hat_function(x, xi_minus, xi, xip1):
    """
    Evaluate left part of hat basis function centered at xi

    Params:
    x:          evaluation points
    xi_minus:   left neighbor node (x_{i-1})
    xi:         center node (x_i)
    xip1:       right neighbor node (x_{i+1})
    """

    h = xi - xi_minus

    if np.isscalar(x):
        if xi_minus <= x <= xi: 
            return (x - xi_minus) / h 
        else:
            return 0.0
    else:
        result = np.zeros_like(x, dtype=float)
        mask = (xi_minus <= x) & (x <= xi)
        result[mask] = (x[mask] -xi_minus) / h 
        return result 

def hat_right(x, xi, xip1):
    """
    Evaluate right part of hat basis function centered at xi 

    Params: 
    x:          evaluation point 
    xi:         center node (x_i)
    xip1:       right neighbor node (x_{i+1})
    """

    h = xip1 - xi 

    if np.isscalar(x):
        if xi <= x <= xip1:
            return (xip1 - x) / h
        else:
            return 0.0
    else: 
        result = np.zeros_like(x, dtype=float)
        mask = (xi <= x) & (x <= xip1)
        result[mask] = (xip1 - x[mask]) / h 
        return result

def assemble_stiffness_matrix():
    pass 

def assemble_load_vector():
    pass 

def dirichlet_application_bc():
    pass 

def solve_system():
    pass

def compute_12_error():
    pass 

def compute_linfinity_error():
    pass 

def compute_l1_error():
    pass 

def get_convergence_order():
    pass 

# Problem 1 


# Problem 2


# Problem 3
