#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


# HELPER FUNCTIONS

def hat_function(x, xi_minus, xi, xip1):
    """
    Evaluate left part of hat basis function centered at xi

    hat basis function Φ_i(x) is piecewise linear
    equals 1 at x=xi and 0 at x=xi-1 and x=xi+1,
    it is non-zero only on [xi-1, xi+1]

    Params:
        x:          evaluation points
        xi_minus:   left neighbor node (x_{i-1})
        xi:         center node (x_i)
        xip1:       right neighbor node (x_{i+1})

    Returns:
        Value of left hat function on interval (x_{i-1}, x_i)
        Formula: (x - x_{i-1}) / h  where h = x_i - x_{i-1}
    """

    h = xi - xi_minus
    if np.isscalar(x):
        if xi_minus <= x < = xi:
            return (x - xi_minus) / h if xi_minus <= x <= xi 
        else:
            return 0.0
    else:
        result = np.zeros_like(x, dtype=float)
        mask = (xi_minus <= x) & (x <= xi)
        result[mask] = (x[mask] - xi_minus) / h
        return result

def hat_right(x, xi, xip1):
    """
    Evaluate right part of hat basis function centered at xi

    Formula: (x_{i+1} - x) / h  where h = x_{i+1} - x_i

    Params:
        x:          evaluation pts 
        xi:         center node (x_i)
        xip1:       right neighbor node (x_{i+1})

    Returns:
        Value of right hat function on interval (x_i, x_{i+1})
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
    Assemble global stiffness matrix A for -u'' = f on (0,1)
    w/ piecewise linear elements on an equidistant grid

    Stiffness matrix entries:
        A[i,j] = ∫_0^1 φ_i'(x) φ_j'(x) dx

    Interior nodes on an equidistant grid with spacing h:
        A[i,i]   = 2/h
        A[i,i-1] = -1/h
        A[i,i+1] = -1/h

    Params:
        n_nodes:     total number of nodes (including boundaries)

    Returns:
        A:          (n_nodes, n_nodes) sparse-style dense matrix (before BCs)
        h:          mesh width
    """

    h = 1.0 / (n_nodes - 1)  # equidistant grid spacing
    A = np.zeros((n_nodes, n_nodes))

    # Loop over all interior nodes (boundaries handled later)
    for i in range(1, n_nodes - 1):
        # Diagonal entry: contribution from left and right elements
        A[i, i] = 2.0 / h
        # Off-diagonal entries: coupling with neighbors
        A[i, i - 1] = -1.0 / h
        A[i, i + 1] = -1.0 / h

    return A, h

def assemble_load_vector(n_nodes, f_func, delta_center=None):
    """
    Assemble global load vector F for -u'' = f on (0,1)

    load vector entries:
        F[i] = ∫_0^1 f(x) φ_i(x) dx

    Use 2-point Gauss-Legendre quadrature on each element to compute
    integrals. The quadrature is exact for polynomials up to degree 3,
    which is higher than the linear FEM approximation (degree 1)

    Special handling for Dirac delta distribution:
        If delta_center is given, integral ∫ δ(x - x_c) φ_i(x) dx
        evals to φ_i(x_c) when x_c coincides with a node, or is
        approximated by placing the delta at nearest node.

    Params:
        n_nodes:        total number of nodes
        f_func:         function f(x) that returns the RHS value at x
        delta_center:   optional center of the Dirac delta (e.g., 0.5)

    Returns:
        F:              (n_nodes,) load vector
    """

    h = 1.0 / (n_nodes - 1)
    F = np.zeros(n_nodes)

    if delta_center is != None:
        # For Dirac delta, F[i] = φ_i(delta_center)
        # delta picks out value of basis function at x_c.
        # eval φ_i at delta_center directly.
        for i in range(1, n_nodes - 1):
            # Find element containing delta_center
            # delta_center is between node k and k+1 where k = int(delta_center / h)
            k = int(delta_center / h)
            if k < 0:
                k = 0
            if k >= n_nodes - 1:
                k = n_nodes - 2

            xi_minus = k * h      # x_{i-1}
            xi = (k + 1) * h      # x_i  (node at delta_center if exact)
            xip1 = (k + 2) * h    # x_{i+1}

            # Checks if delta_center coincides with node i
            if np.isclose(delta_center, i * h, atol=1e-12):
                # φ_i(x_c) = 1 if delta is exactly at node i
                F[i] = 1.0
            elif k == i - 1:
                # delta in element [x_{i-1}, x_i], use left hat
                F[i] = hat_right(delta_center, xi - h, xi)  # wrong, use left
                # Actually: φ_i on left element is (x - x_{i-1})/h
                F[i] = (delta_center - (i - 1) * h) / h
            elif k == i:
                # delta in element [x_i, x_{i+1}], use right hat
                F[i] = (xip1 - delta_center) / h
            # else: φ_i(delta_center) = 0

        # Multiply by strength of the delta (2 for this hw)
        F *= 2.0
        return F

    # Standard case: numerical quadrature on each element
    # 2-point Gauss-Legendre quadrature nodes and weights on [-1, 1]
    gauss_nodes = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    gauss_weights = np.array([1.0, 1.0])

    # Loop over all elements (an element connects node i-1 and node i)
    for el in range(1, n_nodes):
        xi_minus = (el - 1) * h  # x_{el-1}
        xi = el * h              # x_{el}

        # For each node that participates element (el-1 and el), computes the contribution to F[i] from this element
        for node_idx in [el - 1, el]:
            total = 0.0
            for g in range(2):
                # Map Gauss node from [-1, 1] to [x_{el-1}, x_{el}]
                x_local = 0.5 * (xi_minus + xi) + 0.5 * (xi - xi_minus) * gauss_nodes[g]
                # Jacobian of mapping: dx/dξ = h/2
                jacobian = h / 2.0
                # Eval f at the quadrature point
                f_val = f_func(x_local)
                # Evaluate basis function φ_{node_idx} at x_local
                # the basis function is either right part of φ_{el-1} or left part of φ_{el}.
                if node_idx == el - 1:
                    phi_val = hat_right(x_local, xi_minus, xi)
                else:
                    phi_val = hat_function(x_local, xi_minus, xi, (el + 1) * h)
                # Accumulate integral
                total += gauss_weights[g] * f_val * phi_val * jacobian
            F[node_idx] += total

    return F


def apply_dirichlet_bc(A, F, n_nodes):
    """
    Apply homogeneous Dirichlet boundary conditions u(0)=0, u(1)=0.

    For homogeneous BCs, we simply set the first and last rows of A<math> \right\rangle </math>
    to enforce A[0,0]=1, A[-1,-1]=1 (identity rows) and F[0]=F[-1]=0.

    Parameters:
        A:          stiffness matrix (n_nodes, n_nodes)
        F:          load vector (n_nodes,)
        n_nodes:    total number of nodes

    Returns:
        A_bc:       modified stiffness matrix
        F_bc:       modified load vector
    """
    A_bc = A.copy()
    F_bc = F.copy()

    # Set boundary rows to identity
    A_bc[0, :] = 0.0
    A_bc[0, 0] = 1.0
    A_bc[-1, :] = 0.0
    A_bc[-1, -1] = 1.0

    # Set boundary values in F (homogeneous BCs, so 0)
    F_bc[0] = 0.0
    F_bc[-1] = 0.0

    return A_bc, F_bc


def solve_system(A_bc, F_bc):
    """
    Solve linear system A x = F using scipy
    Params:
        A_bc:       modified stiffness matrix
        F_bc:       modified load vector

    Returns:
        u:          numerical solution vector
    """
    u = solve(A_bc, F_bc)
    return u

def compute_l2_error(u_exact, u_numerical, n_nodes):
    """
    Compute the L^2 error: ||u - u_h||_{L^2} = sqrt(integral_0^1 (u(x) - u_h(x))^2 dx)

    Approx integral using trapezoidal rule, second-order accurate
    sufficient for piecewise linear elements

    Params:
        u_exact:        exact solution evaluated at all nodes
        u_numerical:    numerical solution at all nodes
        n_nodes:        number of nodes

    Returns:
        l2_error:       scalar L^2 error
    """

    h = 1.0 / (n_nodes - 1)
    error_squared = (u_exact - u_numerical) ** 2
    # Trapezoidal rule for integration
    l2_integral = h * (0.5 * error_squared[0] + np.sum(error_squared[1:-1]) + 0.5 * error_squared[-1])
    return np.sqrt(l2_integral)

def compute_linf_error(u_exact, u_numerical):
    """
    Compute the L^infinity error: max |u(x) - u_h(x)| over all nodes

    Params:
        u_exact:        exact solution evaluated at all nodes
        u_numerical:    numerical solution at all nodes

    Returns:
        linf_error:     scalar L^infinity error
    """

    return np.max(np.abs(u_exact - u_numerical))

def compute_l1_error(u_exact, u_numerical, n_nodes):
    """
    Compute the L^1 error: integral_0^1 |u(x) - u_h(x)| dx

    Approximated using trapezoidal rule

    Params:
        u_exact:        exact solution evaluated at all nodes
        u_numerical:    numerical solution at all nodes
        n_nodes:        number of nodes

    Returns:
        l1_error:       scalar L^1 error
    """

    h = 1.0 / (n_nodes - 1)
    abs_error = np.abs(u_exact - u_numerical)
    l1_integral = h * (0.5 * abs_error[0] + np.sum(abs_error[1:-1]) + 0.5 * abs_error[-1])
    return l1_integral

def get_convergence_order(errors):
    """
    Estimate convergence order from list of L^2 errors computed
    on refined meshes

    If errors[i] ~ C * h_i^p:
        p ~ log(errors[i] / errors[i+1]) / log(h_i / h_{i+1})

    For equidistant grids: h_{i+1} = h_i / 2 so:
        p ~ log(errors[i] / errors[i+1]) / log(2)

    Params:
        errors:      list of L^2 errors for decreasing mesh sizes

    Returns:
        orders:      list of estimated convergence orders
    """

    orders = []
    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i + 1] > 0:
            p = np.log(errors[i] / errors[i + 1]) / np.log(2.0)
            orders.append(p)
        else:
            orders.append(np.nan)
    return orders


# PROBLEM 1

print("PROBLEM 1:")

# Define right-hand side function f(x) = 1
f_const = lambda x: np.ones_like(x, dtype=float) if np.isscalar(x) is False else 1.0

# Define analytical solution for f ≡ 1
# u(x) = -0.5 * (x - 0.5)^2 + 1/8
u_exact_const = lambda x: -0.5 * (x - 0.5)**2 + 0.125

# Test with single mesh
n_nodes_test = 21
A, h = assemble_stiffness_matrix(n_nodes_test)
F = assemble_load_vector(n_nodes_test, f_const)
A_bc, F_bc = apply_dirichlet_bc(A, F, n_nodes_test)
u_numerical = solve_system(A_bc, F_bc)

# eval exact solution at nodes
x_nodes = np.linspace(0, 1, n_nodes_test)
u_exact_vals = u_exact_const(x_nodes)

# compute L^2 error
l2_error = compute_l2_error(u_exact_vals, u_numerical, n_nodes_test)
print(f"  n_nodes = {n_nodes_test}, h = {h:.4f}")
print(f"  L² error = {l2_error:.6e}")

# plot numerical and exact solutions
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



# PROBLEM 2:

print("PROBLEM 2:")

# PROBLEM 2a:

print("\n--- Problem 2a:")

# Analytical solution for f_2(x) = 2thesymbol(x − 0.5):
# solution is Green's function for -u'' = symbol(x−0.5) scaled by 2
# For -u'' = blah(x−x_c) with u(0)=u(1)=0
#   u(x) = x(1−x_c) for 0 <= x <= x_c
#   u(x) = x_c(1−x) for x_c <= x <= 1
# Scaled by 2:
#   u(x) = 2x(1−0.5) = x for 0 <= x <= 0.5
#   u(x) = 2(0.5)(1−x) = 1−x for 0.5 <= x <= 1
u_exact_delta = lambda x: np.where(x <= 0.5, x, 1.0 - x)

# solve w Dirac delta
delta_center = 0.5
n_nodes_test2 = 21
A2, h2 = assemble_stiffness_matrix(n_nodes_test2)
F2 = assemble_load_vector(n_nodes_test2, lambda x: np.zeros_like(x), delta_center=delta_center)
A2_bc, F2_bc = apply_dirichlet_bc(A2, F2, n_nodes_test2)
u_numerical2 = solve_system(A2_bc, F2_bc)

# Eval exact solution
x_nodes2 = np.linspace(0, 1, n_nodes_test2)
u_exact_vals2 = u_exact_delta(x_nodes2)

# Compute L^2 error
l2_error2 = compute_l2_error(u_exact_vals2, u_numerical2, n_nodes_test2)
print(f"  n_nodes = {n_nodes_test2}, h = {h2:.4f}")
print(f"  L^2 error = {l2_error2:.6e}")

# Plot hopefully this works idk
plt.figure(figsize=(10, 5))
plt.plot(x_nodes2, u_exact_vals2, 'b-', linewidth=2, label='Exact solution')
plt.plot(x_nodes2, u_numerical2, 'ro', markersize=5, label='FEM solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Problem 2a: f₂(x) = 2δ(x−0.5)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('problem2a_solution.png', dpi=150)
plt.show()
print("  Plot saved as 'problem2a_solution.png'")


# Problem 2b:

print("\nProblem 2b:")

# Analytical solution for f_3(x) = sin(2pix):
# solve -u'' = sin(2pix) with u(0)=u(1)=0.
# guess: u(x) = A sin(2pix). Then u''(x) = -4pi^2 A sin(2pix).
# So -u'' = 4pi^2A sin(2pix) = sin(2pix) => A = 1/(4pi^2)
# Therefore: u(x) = sin(2pix) / (4pi^2)
u_exact_sin = lambda x: np.sin(2 * np.pi * x) / (4 * np.pi**2)

# Solve with f_3(x) = sin(2pix)
f_sin = lambda x: np.sin(2 * np.pi * x)
n_nodes_test3 = 21
A3, h3 = assemble_stiffness_matrix(n_nodes_test3)
F3 = assemble_load_vector(n_nodes_test3, f_sin)
A3_bc, F3_bc = apply_dirichlet_bc(A3, F3, n_nodes_test3)
u_numerical3 = solve_system(A3_bc, F3_bc)

# Eval exact solution
x_nodes3 = np.linspace(0, 1, n_nodes_test3)
u_exact_vals3 = u_exact_sin(x_nodes3)

# Compute L^2 error
l2_error3 = compute_l2_error(u_exact_vals3, u_numerical3, n_nodes_test3)
print(f"  n_nodes = {n_nodes_test3}, h = {h3:.4f}")
print(f"  L^2 error = {l2_error3:.6e}")

# Plot plot plot plot plot 
plt.figure(figsize=(10, 5))
plt.plot(x_nodes3, u_exact_vals3, 'b-', linewidth=2, label='Exact solution')
plt.plot(x_nodes3, u_numerical3, 'ro', markersize=5, label='FEM solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Problem 2b: f₃(x) = sin(2πx)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('problem2b_solution.png', dpi=150)
plt.show()
print("  Plot saved as 'problem2b_solution.png'")



# PROBLEM 3:

print("PROBLEM 3")

# Define range of mesh sizes (aka the num of nodes)
# use n_nodes = 2^k + 1 for k = 2, 3, ..., 9
n_nodes_list = [2**k + 1 for k in range(2, 10)]
print(f"  Mesh sizes (n_nodes): {n_nodes_list}")

# errors storage
errors_const = []
errors_delta = []
errors_sin = []

# store mesh widths (for reference)
h_list = []

for n in n_nodes_list:
    h = 1.0 / (n - 1)
    h_list.append(h)

    # f ≡ 1 
    A, _ = assemble_stiffness_matrix(n)
    F = assemble_load_vector(n, f_const)
    A_bc, F_bc = apply_dirichlet_bc(A, F, n)
    u_num = solve_system(A_bc, F_bc)
    x = np.linspace(0, 1, n)
    u_ex = u_exact_const(x)
    err = compute_l2_error(u_ex, u_num, n)
    errors_const.append(err)

    # f_2(x) = 2delta(x−0.5)
    A_d, _ = assemble_stiffness_matrix(n)
    F_d = assemble_load_vector(n, lambda x: np.zeros_like(x), delta_center=0.5)
    A_d_bc, F_d_bc = apply_dirichlet_bc(A_d, F_d, n)
    u_num_d = solve_system(A_d_bc, F_d_bc)
    u_ex_d = u_exact_delta(x)
    err_d = compute_l2_error(u_ex_d, u_num_d, n)
    errors_delta.append(err_d)

    # f_3(x) = sin(2pix)
    A_s, _ = assemble_stiffness_matrix(n)
    F_s = assemble_load_vector(n, f_sin)
    A_s_bc, F_s_bc = apply_dirichlet_bc(A_s, F_s, n)
    u_num_s = solve_system(A_s_bc, F_s_bc)
    u_ex_s = u_exact_sin(x)
    err_s = compute_l2_error(u_ex_s, u_num_s, n)
    errors_sin.append(err_s)

    print(f"  n={n:4d}, h={h:.4f}, L²(f=1)={err:.6e}, L²(δ)={err_d:.6e}, L²(sin)={err_s:.6e}")

# Compute convergence orders
orders_const = get_convergence_order(errors_const)
orders_delta = get_convergence_order(errors_delta)
orders_sin = get_convergence_order(errors_sin)

print(f"\n  Convergence orders (f=1):     {orders_const}")
print(f"  Convergence orders (f=δ):     {orders_delta}")
print(f"  Convergence orders (f=sin):   {orders_sin}")

# Plot convergence on loglog scale
plt.figure(figsize=(10, 6))
plt.loglog(h_list, errors_const, 'bo-', label=f'f=1 (order ~{np.mean(orders_const):.2f})')
plt.loglog(h_list, errors_delta, 'rs-', label=f'f=2δ (order ~{np.mean(orders_delta):.2f})')
plt.loglog(h_list, errors_sin, 'g^-', label=f'f=sin (order ~{np.mean(orders_sin):.2f})')

# Reference line for O(h^2) convergence
h_ref = np.array([0.01, 0.1])
l2_ref = h_ref**2
plt.loglog(h_ref, l2_ref, 'k--', linewidth=1, label='O(h²) reference')

plt.xlabel('Mesh width h')
plt.ylabel('L^2 error')
plt.title('Convergence Analysis: L^2 Error vs. Mesh Width')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.savefig('convergence_analysis.png', dpi=150)
plt.show()
print("\n  Plot saved as 'convergence_analysis.png'")

# OBSERVATION: Grid point at x = 0.5

print("OBSERVATION:")
print("""
\nWhen a grid point coincides exactly with the Dirac delta center (x=0.5),
the numerical solution captures the kink in the exact solution perfectly.
The basis functions are piecewise linear, and if a node sits at x=0.5,
one element ends there and another begins there, this allows the FEM solution
to represent the discontinuety in u' exactly.

If no grid point is at x=0.5, the delta is smeared across two elements,
and the convergence rate for the delta case might degrade slightly because
the exact solution has a kink between nodes.

This is why you should observe near-perfect O(h^2) convergence for f=1 and
f=sin (smooth RHS), but the delta case may show slightly different behavior
depending on whether a node aligns with x=0.5.
""")
