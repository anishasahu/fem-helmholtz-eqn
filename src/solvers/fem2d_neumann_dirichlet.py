from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import jv, jvp, roots_legendre, yv, yvp

from .base import BaseSolver


class FEM2DNeumannDirichletSolver(BaseSolver):
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

        return -normal_derivative

    def neumann_element_matrices(self, idx):
        # Get element nodes and coordinates
        element = self.eqn.elements[idx]
        node_coords = self.eqn.nodes[element]

        # Initialize element contribution vector
        N_e = np.zeros(4, dtype=complex)
        # 1D Gauss quadrature points and weights
        gauss_points, gauss_weights = roots_legendre(3)

        # Define the edges of the 4-node element
        edges = [
            (0, 1),  # Bottom edge (xi varies, eta = -1)
            (1, 2),  # Right edge (xi = 1, eta varies)
            (2, 3),  # Top edge (xi varies, eta = 1)
            (3, 0),  # Left edge (xi = -1, eta varies)
        ]

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

            # Calculate edge length
            edge_length = np.linalg.norm(node_coords[edge[1]] - node_coords[edge[0]])

            # This edge is on the inner boundary, perform line integration
            for i, xi in enumerate(gauss_points):
                N = self.get_shape_fuctions_1d(xi)

                ds = edge_length / 2

                # Current position
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

    def apply_boundary_conditions(
        self, A: NDArray, b: NDArray, tol: float = 1e-10
    ) -> Tuple[NDArray, NDArray]:
        r = np.linalg.norm(self.eqn.nodes, axis=1)
        outer_nodes = np.abs(r - self.outer_radius) < tol

        # Apply Dirichlet conditions in outer boundary
        # u(r=outer_radius) = 0
        A[outer_nodes] = 0
        A[outer_nodes, outer_nodes] = 1
        b[outer_nodes] = 0

        return A, b

    def assemble(self) -> None:
        self.K = np.zeros((self.eqn.n_nodes, self.eqn.n_nodes), dtype=complex)
        self.F = np.zeros(self.eqn.n_nodes, dtype=complex)
        self.N_global = np.zeros(self.eqn.n_nodes, dtype=complex)

        for idx in self.eqn.inner_boundary_element_indices:
            N_e = self.neumann_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(4):
                self.N_global[global_indices[i]] += N_e[i]

        for idx in range(self.eqn.n_elements):
            K_e, F_e = self.get_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(4):
                for j in range(4):
                    self.K[global_indices[i], global_indices[j]] += K_e[i, j]

                self.F[global_indices[i]] += F_e[i]

        self.F += self.N_global 

    def solve(self) -> Tuple[NDArray, NDArray]:
        self.assemble()

        self.K, self.F = self.apply_boundary_conditions(self.K, self.F)
        u_complex = np.linalg.solve(self.K, self.F)
        u_real = np.real(u_complex)
        u_imag = np.imag(u_complex)

        return u_real, u_imag

    def get_analytical_solution(self, x, y) -> complex:
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
    # def get_analytical_solution(self, x, y) -> complex:
    #     r = np.sqrt(x**2 + y**2)
    #     # theta = np.arctan2(y, x)
    #     k = np.sqrt(self.k_squared)

    #     A0 = 1/hankel1(0, k * self.outer_radius)

    #     # Compute the exact solution
    #     u_ex = A0 * hankel1(0, k*r)

    #     return u_ex

