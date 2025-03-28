from typing import Tuple

import numpy as np
from scipy.special import jv, jvp, roots_legendre, yv, yvp

from src.helmholtz import HelmHoltz
from src.utils import timeit


class FEM2DNeumannDirichletSolver:
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

    def get_normal_derivative(self, x, y) -> complex:
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        k = np.sqrt(self.k_squared)

        # Initialize normal derivative
        normal_derivative = complex(0.0, 0.0)

        for m in range(-self.n_fourier, self.n_fourier + 1):
            # Derivative of Bessel function
            jnp = jvp(m, k * r)

            # Contribution for n
            normal_derivative += (1j**m) * k * jnp * np.exp(1j * m * theta)

        return normal_derivative

    def neumann_element_matrices(self, idx):
        # Get element nodes and coordinates
        element = self.eqn.elements[idx]
        node_coords = self.eqn.nodes[element]

        # Initialize element contribution vector
        N_e = np.zeros(4, dtype=complex)

        # Define the edges of the 4-node element
        edges = [
            (0, 1),  # Bottom edge (xi varies, eta = -1)
            (1, 2),  # Right edge (xi = 1, eta varies)
            (2, 3),  # Top edge (xi varies, eta = 1)
            (3, 0),  # Left edge (xi = -1, eta varies)
        ]

        # 1D Gauss quadrature points and weights
        gauss_points, gauss_weights = roots_legendre(2)

        # Check each edge to see if it lies on the inner boundary
        for edge in edges:
            # Check if both nodes in this edge are on the inner boundary
            if not (
                np.isclose(np.linalg.norm(node_coords[edge[0]]), self.eqn.inner_radius)
                and np.isclose(
                    np.linalg.norm(node_coords[edge[1]]), self.eqn.inner_radius
                )
            ):
                continue

            edge_length = np.sqrt(
                (node_coords[edge[1], 0] - node_coords[edge[0], 0]) ** 2
                + (node_coords[edge[1], 1] - node_coords[edge[0], 1]) ** 2
            )

            # This edge is on the inner boundary, perform line integration
            for i, xi in enumerate(gauss_points):
                N = self.get_shape_fuctions_1d(xi)

                ds = edge_length / 2

                # # Current position
                current_pos = np.zeros(2)
                for k in range(2):  # 2 nodes per edge
                    current_pos[0] += N[k] * node_coords[edge[k], 0]
                    current_pos[1] += N[k] * node_coords[edge[k], 1]
                x, y = current_pos

                # Get the normal derivative using the function
                normal_derivative = self.get_normal_derivative(x, y)

                # Weight for this Gauss point
                weight = gauss_weights[i]

                # Add contribution to element vector
                for local_m in range(2):  # Only iterate over 1D shape functions
                    global_m = edge[local_m]  # Map to the correct global node index
                    N_e[global_m] += N[local_m] * normal_derivative * ds * weight

        return N_e

    def apply_boundary_conditions(self, A, b, tol: float = 1e-10):
        r = np.linalg.norm(self.eqn.nodes, axis=1)
        outer_nodes = np.abs(r - self.outer_radius) < tol

        # Apply Dirichlet conditions in outer boundary
        # u(r=2) = 0
        A[outer_nodes] = 0
        A[outer_nodes, outer_nodes] = 1
        b[outer_nodes] = 0

        return A, b

    def assemble(self) -> None:
        self.K = np.zeros((self.eqn.n_nodes, self.eqn.n_nodes), dtype=complex)
        self.F = np.zeros(self.eqn.n_nodes, dtype=complex)
        self.N = np.zeros(self.eqn.n_nodes, dtype=complex)

        for idx in self.eqn.inner_boundary_element_indices:
            N_e = self.neumann_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(4):
                self.N[global_indices[i]] += N_e[i]

        for idx in range(self.eqn.n_elements):
            K_e, F_e = self.get_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(4):
                for j in range(4):
                    self.K[global_indices[i], global_indices[j]] += K_e[i, j]

                self.F[global_indices[i]] += F_e[i]

        self.F += self.N

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
        theta = np.arctan2(y, x)
        k = np.sqrt(self.k_squared)

        # Compute the exact solution
        u_ex = 0 + 0j

        for m in range(-self.n_fourier, self.n_fourier + 1):
            u_ex += (
                1j**m
                * np.exp(1j * m * theta)
                * jvp(m, k)
                * (
                    jv(m, self.outer_radius * k) * yv(m, k * r)
                    - yv(m, self.outer_radius * k) * jv(m, k * r)
                )
                / (
                    jvp(m, k) * yv(m, self.outer_radius * k)
                    - yvp(m, k) * jv(m, self.outer_radius * k)
                )
            )

        return u_ex
