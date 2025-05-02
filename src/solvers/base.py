from src.helmholtz import HelmHoltz
from abc import ABC

from typing import Tuple
from scipy.special import roots_legendre


import numpy as np

__all__ = ["BaseSolver"]


class BaseSolver(ABC):
    def __init__(
        self,
        type: str,
        eqn: "HelmHoltz",
        k_squared: float = 10.0,
        n_fourier: int = 1,
        abc_order: int = 3,
        inner_radius: float = 1.0,
        outer_radius: float = 1.5,
    ):
        self.type = type
        self.eqn = eqn
        self.k_squared = k_squared
        self.n_fourier = n_fourier
        self.abc_order = abc_order
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def get_shape_functions(self, xi, eta):
        # Shape functions for 4-node element
        N = np.zeros(4)
        dN_dxi = np.zeros(4)
        dN_deta = np.zeros(4)

        # Shape functions
        N[0] = 0.25 * (1 - xi) * (1 - eta)
        N[1] = 0.25 * (1 + xi) * (1 - eta)
        N[2] = 0.25 * (1 + xi) * (1 + eta)
        N[3] = 0.25 * (1 - xi) * (1 + eta)

        # Derivatives of shape functions
        dN_dxi[0] = -0.25 * (1 - eta)
        dN_dxi[1] = 0.25 * (1 - eta)
        dN_dxi[2] = 0.25 * (1 + eta)
        dN_dxi[3] = -0.25 * (1 + eta)

        dN_deta[0] = -0.25 * (1 - xi)
        dN_deta[1] = -0.25 * (1 + xi)
        dN_deta[2] = 0.25 * (1 + xi)
        dN_deta[3] = 0.25 * (1 - xi)

        return N, dN_dxi, dN_deta

    def get_shape_fuctions_1d(self, xi):
        N = np.zeros(2)

        N[0] = 0.5 * (1 - xi)
        N[1] = 0.5 * (1 + xi)

        return N

    def get_element_matrices(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        element_nodes = self.eqn.elements[idx]
        node_coords = self.eqn.nodes[element_nodes]

        K_e = np.zeros((4, 4), dtype=complex)
        F_e = np.zeros(4, dtype=complex)

        gauss_points, gauss_weights = roots_legendre(2)

        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                # Shape functions and derivatives at this Gauss point
                N, dN_dxi, dN_deta = self.get_shape_functions(xi, eta)

                # Jacobian matrix
                J = np.zeros((2, 2))
                for k in range(4):
                    J[0, 0] += dN_dxi[k] * node_coords[k, 0]
                    J[0, 1] += dN_dxi[k] * node_coords[k, 1]
                    J[1, 0] += dN_deta[k] * node_coords[k, 0]
                    J[1, 1] += dN_deta[k] * node_coords[k, 1]

                # Determinant of Jacobian
                detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

                # Inverse of Jacobian
                Jinv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ

                # Derivatives of shape functions with respect to x and y
                dN_dx = np.zeros(4)
                dN_dy = np.zeros(4)

                for k in range(4):
                    dN_dx[k] = Jinv[0, 0] * dN_dxi[k] + Jinv[0, 1] * dN_deta[k]
                    dN_dy[k] = Jinv[1, 0] * dN_dxi[k] + Jinv[1, 1] * dN_deta[k]

                # Weight for this Gauss point
                weight = gauss_weights[i] * gauss_weights[j]

                # Add contribution to element matrices
                for m in range(4):
                    for n in range(4):
                        # Stiffness and mass matrices
                        K_e[m, n] += (
                            weight
                            * detJ
                            * (
                                -(dN_dx[m] * dN_dx[n] + dN_dy[m] * dN_dy[n])
                                + self.k_squared * N[m] * N[n]
                            )
                        )

        return K_e, F_e

    def compute_L2_error(self, u_num_real, u_num_imag):
        L2_error = 0.0

        # Get Gauss quadrature points and weights
        gauss_points, gauss_weights = roots_legendre(2)

        # Iterate over all elements
        for idx in range(self.eqn.n_elements):
            element_nodes = self.eqn.elements[idx]
            node_coords = self.eqn.nodes[element_nodes]

            # Get numerical solution at this element's nodes
            u_elem_real = u_num_real[element_nodes]
            u_elem_imag = u_num_imag[element_nodes]
            u_elem = u_elem_real + 1j * u_elem_imag

            element_error = 0.0

            # Numerical integration over the element
            for i, xi in enumerate(gauss_points):
                for j, eta in enumerate(gauss_points):
                    # Shape functions at this Gauss point
                    N, dN_dxi, dN_deta = self.get_shape_functions(xi, eta)

                    # Calculate x, y coordinates at this Gauss point
                    x = 0.0
                    y = 0.0
                    for k in range(4):
                        x += N[k] * node_coords[k, 0]
                        y += N[k] * node_coords[k, 1]

                    # Interpolate numerical solution at this point
                    u_num = 0.0
                    for k in range(4):
                        u_num += N[k] * u_elem[k]

                    # Analytical solution at this point
                    if self.type == "dirichlet_sommerfeld" or self.type == "default":
                        u_exact = self.get_analytical_solution_ordered(x, y, self.abc_order)
                    else:
                        u_exact = self.get_analytical_solution(x, y)

                    # Jacobian for coordinate transformation
                    J = np.zeros((2, 2))
                    for k in range(4):
                        J[0, 0] += dN_dxi[k] * node_coords[k, 0]
                        J[0, 1] += dN_dxi[k] * node_coords[k, 1]
                        J[1, 0] += dN_deta[k] * node_coords[k, 0]
                        J[1, 1] += dN_deta[k] * node_coords[k, 1]

                    # Determinant of Jacobian
                    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

                    # Contribution to error at this Gauss point
                    error_at_point = abs(u_num - u_exact) ** 2
                    element_error += (
                        error_at_point * detJ * gauss_weights[i] * gauss_weights[j]
                    )

            L2_error += element_error

        return np.sqrt(L2_error)

    def assemble(self):
        raise NotImplementedError("Assembly not implemented")

    def solve(self):
        raise NotImplementedError("Solver not implemented")

    def get_analytical_solution(self, x, y):
        raise NotImplementedError("Analytical solution not implemented")

    def get_analytical_solution_ordered(self, x, y, order):
        raise NotImplementedError("Analytical solution not implemented")
