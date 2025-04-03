from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0, y0

from src.utils import timeit
from src.solvers.base import BaseSolver


class FEM2DDirichletSolver(BaseSolver):
    def apply_boundary_conditions(
        self, A: NDArray, b: NDArray, tol: float = 1e-10
    ) -> Tuple[NDArray, NDArray]:
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

        # (@anishasahu) Something missing here?

        for idx in range(self.eqn.n_elements):
            K_e, F_e = self.get_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(4):
                for j in range(4):
                    self.K[global_indices[i], global_indices[j]] += K_e[i, j]

                self.F[global_indices[i]] += F_e[i]

    @timeit
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
        return A * j0(k * r) + B * y0(k * r)
