from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0, y0, jv, yv, roots_legendre

from .base import BaseSolver


class FEM2DDirichletSommerfeldSolver(BaseSolver):
    def apply_boundary_conditions(
        self, A: NDArray, b: NDArray, tol: float = 1e-10
    ) -> Tuple[NDArray, NDArray]:
        r = np.linalg.norm(self.eqn.nodes, axis=1)
        inner_nodes = np.abs(r - self.inner_radius) < tol

        # Apply Dirichlet conditions
        # u(r=1) = 1
        A[inner_nodes] = 0
        A[inner_nodes, inner_nodes] = 1
        b[inner_nodes] = 1

        return A, b

    def sommerfeld_element_matrices(self, idx):
        element = self.eqn.elements[idx]
        node_coords = self.eqn.nodes[element]

        S_e = np.zeros((4, 4), dtype=complex)
        gauss_points, gauss_weights = roots_legendre(2)

        # Define the edges of the 4-node element
        edges = [
            (0, 1),  # Bottom edge (xi varies, eta = -1)
            (1, 2),  # Right edge (xi = 1, eta varies)
            (2, 3),  # Top edge (xi varies, eta = 1)
            (3, 0),  # Left edge (xi = -1, eta varies)
        ]

        # Check each edge to see if it lies on the outer boundary
        for edge in edges:
            # Check if both nodes in this edge are on the outer boundary
            if not (
                np.isclose(np.linalg.norm(node_coords[edge[0]]), self.eqn.outer_radius)
                and np.isclose(
                    np.linalg.norm(node_coords[edge[1]]), self.eqn.outer_radius
                )
            ):
                continue

            edge_length = np.linalg.norm(node_coords[edge[1]] - node_coords[edge[0]])

            # This edge is on the outer boundary, perform line integration
            for i, xi in enumerate(gauss_points):
                N = self.get_shape_fuctions_1d(xi)

                ds = edge_length / 2

                # Sommerfeld coefficient (typically i*k for 2D Helmholtz)
                sommerfeld_coef = self.eqn.get_abc_coeff(
                    k_squared=self.k_squared, order=self.abc_order
                )

                # Weight for this Gauss point
                weight = gauss_weights[i]

                # Distribute the 1D shape functions into the full 4-node system
                for local_m in range(2):  # Only iterate over 1D shape functions
                    global_m = edge[local_m]  # Map to the correct global node index
                    for local_n in range(2):
                        global_n = edge[local_m]
                        S_e[global_m, global_n] += (
                            N[local_m] * N[local_n] * sommerfeld_coef * ds * weight
                        )

        return S_e

    def assemble(self) -> None:
        self.K = np.zeros((self.eqn.n_nodes, self.eqn.n_nodes), dtype=complex)
        self.F = np.zeros(self.eqn.n_nodes, dtype=complex)
        self.S = np.zeros((self.eqn.n_nodes, self.eqn.n_nodes), dtype=complex)

        for idx in self.eqn.outer_boundary_element_indices:
            S_e = self.sommerfeld_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(4):
                for j in range(4):
                    self.S[global_indices[i], global_indices[j]] += S_e[i, j]

        for idx in range(self.eqn.n_elements):
            K_e, F_e = self.get_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(4):
                for j in range(4):
                    self.K[global_indices[i], global_indices[j]] += K_e[i, j]

                self.F[global_indices[i]] += F_e[i]

        self.K += self.S

    def solve(self) -> Tuple[NDArray, NDArray]:
        self.assemble()

        self.K, self.F = self.apply_boundary_conditions(self.K, self.F)
        u_complex = np.linalg.solve(self.K, self.F)
        u_real = np.real(u_complex)
        u_imag = np.imag(u_complex)

        return u_real, u_imag

    def get_analytical_solution(self, x, y):
        r = np.sqrt(x**2 + y**2)
        k = np.sqrt(self.k_squared)

        # Boundary conditions: u(inner_radius) = 1, u(outer_radius) = 0
        u1 = 1
        u2 = 0

        # Construct the coefficient matrix using Bessel functions
        matrix = np.array(
            [
                [
                    j0(k),
                    y0(k),
                ],  # Inner boundary condition
                [
                    -jv(1, k * self.outer_radius) - 1j * j0(k * self.outer_radius),
                    -yv(1, k * self.outer_radius) - 1j * y0(k * self.outer_radius),
                ],  # Outer boundary condition
            ]
        )

        # Right-hand side vector for boundary conditions
        rhs = np.array([u1, u2])

        # Solve the system of equations to find coefficients A and B
        A, B = np.linalg.solve(matrix, rhs)

        # Compute and return the analytical solution at all node positions
        return A * j0(k * r) + B * y0(k * r)
