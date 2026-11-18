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

def assemble_stiffness_matrix(n_nodes):
    """
    Assembles the Global Stiffness Matrix A for -u''=f 
    on (0,1) w/ piecewise linear elements on equidistant grid

    Params:
    n_nodes:        total number of nodes (includes boundaries)
    """
    
    h = 1.0 / (n_nodes - 1) # equidistant grid spacing
    A = np.zeros((n_nodes, n_nodes))

    for i in range(1, n_nodes - 1):
        A[i, i] = 2.0 / h
        A[i, i - 1] = -1.0 / h
        A[i, i + 1] = -1.0 / h
    return A, h  

def assemble_load_vector(n_nodes, f_func, delta_center=None):
    """
    Params:
    n_nodes:        Total Number of nodes
    f_func:         function f(x) that returns right hand side value at x 
    delta_center:   option center of dirac delta 
    """
    h = 1.0 / (n_nodes - 1)
    F = np.zeros(n_nodes)

    if delta_center != None:
        for i in range(1, n_nodes - 1):
            k = int(delta_center / h)
            if k < 0:
                k = 0
            if k >= n_nodes - 1:
                k = n_nodes - 2

            xi_minus = k * h 
            xi = (k + 1) * h 
            xip1 = (k + 2) * h 

            if np.isclose(delta_center, i * h, atol=1e-12):
                F[i] = 1.0
            elif k == i - 1:
                F[i] = hat_right(delta_center, xi - h, xi)
                F[i] = (delta_center - (i - 1) * h) / h 
            elif k == i:
                F[i] = (xip1 - delta_center) / h 

        F *= 2.0
        return F 
    
    gauss_nodes = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    gauss_weights = np.array([1.0, 1.0])

    for el in range(1, n_nodes):
        xi_minus = (el - 1) * h 
        xi = el * h 

        for node_idx in [el - 1, el]:
            total = 0.0
            for g in range(2):
                x_local = 0.5 * (xi_minus + xi) + 0.5 * (xi - xi_minus) * gauss_nodes[g]
                jacobian = h / 2.0
                f_val = f_func(x_local)

                if node_idx ==  el - 1:
                    phi_val = hat_right(x_local, xi_minus, xi)
                else: 
                    phi_val = hat_function(x_local, xi_minus, xi, (el + 1) * h)

                total += gauss_weights[g] * f_val * phi_val * jacobian
            F[node_idx] += total 
    return F 

def dirichlet_application_bc(A, F, n_nodes):
    A_bc = A.copy()
    F_bc = F.copy()

    A_bc[0, :]   = 0.0
    A_bc[0, 0]   = 1.0
    A_bc[-1, :]  = 0.0
    A_bc[-1, -1] = 1.0
    
    F_bc[0] = 0.0
    F_bc[-1] = 0.0

    return A_bc, F_bc

def solve_system(A_bc, F_bc):
    u = solve(A_bc, F_bc)
    return u 

def compute_12_error(u_exact, u_numerical, n_nodes):
    h = 1.0 / (n_nodes - 1)
    error_squared = (u_exact - u_numerical) ** 2 
    
    l2_integral = h * (0.5 * error_squared[0] + np.sum(error_squared[1:-1]) + 0.5 * error_squared[-1])
    return np.sqrt(l2_integral)

def compute_linfinity_error(u_exact, u_numerical):
    return np.max(np.abs(u_exact - u_numerical))

def compute_l1_error(u_exact, u_numerical, n_nodes):
    h = 1.0 / (n_nodes - 1)
    abs_error = np.abs(u_exact - u_numerical)
    l1_integral = h * (0.5 * abs_error[0] + np.sum(abs_error[1:-1]) + 0.5 * abs_error[-1])
    return l1_integral

def get_convergence_order(errors):
    orders = []
    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i + 1] > 0:
            p = np.log(errors[i] / errors[i + 1]) / np.log(2.0)
            orders.append(p)
        else:
            orders.append(np.nan)
    return orders 

# Problem 1 
print("PROBLEM 1")
print("=" * 75)

f_const = lambda x: np.ones_like(x, dtype=float) if np.isscalar(x) is False else 1.0

u_exact_const = lambda x: -0.5 * (x - 0.5)**2 + 0.125

n_nodes_test = 21
A, h = assemble_stiffness_matrix(n_nodes_test)
F = assemble_load_vector(n_nodes_test, f_const)
A_bc, Fbc = dirichlet_application_bc(A, F, n_nodes_test)
u_numerical = solve_system(A_bc, F_bc)

x_nodes = np.linspace(0, 1, n_nodes_test)
u_exact_vals = u_exact_const(x_nodes)

l2_error = compute_12_error(u_exact_vals, u_numerical, n_nodes_test)
print(f" n_nodes = {n_nodes_test}, h = {h:.4f}")
print(f" L^2 error = {l2_error:.6e}")

plt.figure(figsize=(10, 5))
plt.plot(x_nodes, u_exact_vals, 'b-', linewidth=2, label='Exact solution')
plt.plot(x_nodes, u_numerical, 'ro', markersize=5, label='FEM solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Problem 1: f ≡ 1 — Numerical vs Exact Solution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('problem1_solution.png', dpi=150)
plt.show()
print("  Plot saved as 'problem1_solution.png'")

# Problem 2


# Problem 3
