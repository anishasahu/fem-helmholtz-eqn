from typing import Tuple

import numpy as np
from scipy.special import j0, roots_legendre, y0

from src.helmholtz import HelmHoltz
from src.utils import timeit


class FEM2DDirichletSolver:
    def __init__(
        self,
        eqn: "HelmHoltz",
        k_squared: float = 10.0,
        n_fourier: int = 50,
        abc_order: int = 1,
        inner_radius: float = 1.0,
        outer_radius: float = 1.5,
    ):
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

    def apply_boundary_conditions(self, A, b, tol: float = 1e-10):
        r = np.linalg.norm(self.eqn.nodes, axis=1)
        inner_nodes = np.abs(r - self.inner_radius) < tol
        outer_nodes = np.abs(r - self.outer_radius) < tol

        # Apply Dirichlet conditions
        # u(r=1) = 0
        A[inner_nodes] = 0
        A[inner_nodes, inner_nodes] = 1
        b[inner_nodes] = 1

        # u(r=2) = 1
        A[outer_nodes] = 0
        A[outer_nodes, outer_nodes] = 1
        b[outer_nodes] = 0

        return A, b

    def assemble(self) -> None:
        self.K = np.zeros((self.eqn.n_nodes, self.eqn.n_nodes), dtype=complex)
        self.F = np.zeros(self.eqn.n_nodes, dtype=complex)

        # Something missing here

        for idx in range(self.eqn.n_elements):
            K_e, F_e = self.get_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(4):
                for j in range(4):
                    self.K[global_indices[i], global_indices[j]] += K_e[i, j]

                self.F[global_indices[i]] += F_e[i]

    @timeit
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        self.assemble()

        self.K, self.F = self.apply_boundary_conditions(self.K, self.F)
        u_complex = np.linalg.solve(self.K, self.F)
        u_real = np.real(u_complex)
        u_imag = np.imag(u_complex)

        return u_real, u_imag

    def get_analytical_solution(self, x, y):
        r = np.sqrt(x**2 + y**2)
        k = np.sqrt(self.k_squared)

        u1 = 1
        u2 = 0

        # Construct the coefficient matrix using Bessel functions
        matrix = np.array(
            [
                [
                    j0(k * self.inner_radius),
                    y0(k * self.inner_radius),
                ],  # Inner boundary condition
                [
                    j0(k * self.outer_radius),
                    y0(k * self.outer_radius),
                ],  # Outer boundary condition
            ]
        )

        # Right-hand side vector for boundary conditions
        rhs = np.array([u1, u2])

        # Solve the system of equations to find coefficients A and B
        A, B = np.linalg.solve(matrix, rhs)

        # Compute and return the analytical solution at all node positions
        return A * j0(k * r) + B * y0(k* r)
