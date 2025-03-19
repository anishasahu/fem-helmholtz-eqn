from typing import Tuple

import numpy as np
from scipy.special import h1vp, hankel1, jvp, roots_legendre

from src.helmholtz import HelmHoltz


class FEM2DSolver:
    def __init__(
        self,
        eqn: "HelmHoltz",
        k_squared: float = 1.0,
        n_fourier: int = 10,
        abc_order: int = 3,
    ):
        self.eqn = eqn
        self.k_squared = k_squared
        self.n_fourier = n_fourier
        self.abc_order = abc_order

    def get_shape_functions(self, xi, eta):
        # Shape functions for 9-node element
        N = np.zeros(9)
        dN_dxi = np.zeros(9)
        dN_deta = np.zeros(9)

        # Corner nodes
        N[0] = 0.25 * xi * eta * (xi - 1) * (eta - 1)
        N[2] = 0.25 * xi * eta * (xi + 1) * (eta - 1)
        N[4] = 0.25 * xi * eta * (xi + 1) * (eta + 1)
        N[6] = 0.25 * xi * eta * (xi - 1) * (eta + 1)

        # Mid-side nodes
        N[1] = 0.5 * eta * (eta - 1) * (1 - xi**2)
        N[3] = 0.5 * xi * (xi + 1) * (1 - eta**2)
        N[5] = 0.5 * eta * (eta + 1) * (1 - xi**2)
        N[7] = 0.5 * xi * (xi - 1) * (1 - eta**2)

        # Center node
        N[8] = (1 - xi**2) * (1 - eta**2)

        # Derivatives of shape functions
        # Corner nodes
        dN_dxi[0] = 0.25 * eta * (eta - 1) * (2 * xi - 1)
        dN_dxi[2] = 0.25 * eta * (eta - 1) * (2 * xi + 1)
        dN_dxi[4] = 0.25 * eta * (eta + 1) * (2 * xi + 1)
        dN_dxi[6] = 0.25 * eta * (eta + 1) * (2 * xi - 1)

        dN_deta[0] = 0.25 * xi * (xi - 1) * (2 * eta - 1)
        dN_deta[2] = 0.25 * xi * (xi + 1) * (2 * eta - 1)
        dN_deta[4] = 0.25 * xi * (xi + 1) * (2 * eta + 1)
        dN_deta[6] = 0.25 * xi * (xi - 1) * (2 * eta + 1)

        # Mid-side nodes
        dN_dxi[1] = -xi * eta * (eta - 1)
        dN_dxi[3] = 0.5 * (1 - eta**2) * (2 * xi + 1)
        dN_dxi[5] = -xi * eta * (eta + 1)
        dN_dxi[7] = 0.5 * (1 - eta**2) * (2 * xi - 1)

        dN_deta[1] = 0.5 * (1 - xi**2) * (2 * eta - 1)
        dN_deta[3] = -eta * xi * (xi + 1)
        dN_deta[5] = 0.5 * (1 - xi**2) * (2 * eta + 1)
        dN_deta[7] = -eta * xi * (xi - 1)

        # Center node
        dN_dxi[8] = -2 * xi * (1 - eta**2)
        dN_deta[8] = -2 * eta * (1 - xi**2)

        return N, dN_dxi, dN_deta

    def get_element_matrices(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        element_nodes = self.eqn.elements[idx]
        node_coords = self.eqn.nodes[element_nodes]

        K_e = np.zeros((9, 9), dtype=complex)
        F_e = np.zeros(9, dtype=complex)

        gauss_points, gauss_weights = roots_legendre(3)

        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                # Shape functions and derivatives at this Gauss point
                N, dN_dxi, dN_deta = self.get_shape_functions(xi, eta)

                # Jacobian matrix
                J = np.zeros((2, 2))
                for k in range(9):
                    J[0, 0] += dN_dxi[k] * node_coords[k, 0]
                    J[0, 1] += dN_dxi[k] * node_coords[k, 1]
                    J[1, 0] += dN_deta[k] * node_coords[k, 0]
                    J[1, 1] += dN_deta[k] * node_coords[k, 1]

                # Determinant of Jacobian
                detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

                # Inverse of Jacobian
                Jinv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ

                # Derivatives of shape functions with respect to x and y
                dN_dx = np.zeros(9)
                dN_dy = np.zeros(9)

                for k in range(9):
                    dN_dx[k] = Jinv[0, 0] * dN_dxi[k] + Jinv[0, 1] * dN_deta[k]
                    dN_dy[k] = Jinv[1, 0] * dN_dxi[k] + Jinv[1, 1] * dN_deta[k]

                # Weight for this Gauss point
                weight = gauss_weights[i] * gauss_weights[j] * detJ

                # Add contribution to element matrices
                for m in range(9):
                    for n in range(9):
                        # Stiffness matrix: (∇φm·∇φn - k²φmφn)
                        K_e[m, n] += weight * (
                            -(dN_dx[m] * dN_dx[n] + dN_dy[m] * dN_dy[n])
                            + self.k_squared * N[m] * N[n]
                        )

        return K_e, F_e

    def get_normal_derivative(self, x, y) -> complex:
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        k = np.sqrt(self.k_squared)

        # Initialize flux
        flux = complex(0.0, 0.0)

        for m in range(-self.n_fourier, self.n_fourier + 1):
            # Derivative of Bessel function
            jnp = jvp(m, k * r)

            # Contribution for positive n
            flux += (self.eqn.i**m) * k * np.exp(self.eqn.i * m * theta) * jnp

        return flux

    def abc_condition(self, order: int):
        k = np.sqrt(self.k_squared)

        # set default value
        coef = self.eqn.i * k

        if order == 1:
            coef = self.eqn.i * k
        elif order == 2:
            coef = -self.eqn.i * k - 1.0 / (2.0 * self.eqn.outer_radius)
        elif order == 3:
            for i, ith_node in enumerate(self.eqn.outer_boundary_node_indices):
                theta_i = np.arctan2(
                    self.eqn.nodes[ith_node, 1], self.eqn.nodes[ith_node, 0]
                )
                for j, jth_node in enumerate(self.eqn.outer_boundary_node_indices):
                    theta_j = np.arctan2(
                        self.eqn.nodes[jth_node, 1], self.eqn.nodes[jth_node, 0]
                    )
                    if abs(theta_i - theta_j) < 0.5:
                        coef = (
                            -self.eqn.i * k
                            - 1.0 / (2.0 * self.eqn.outer_radius)
                            - self.eqn.i / (8.0 * k * self.eqn.outer_radius**2)
                        )
        else:
            raise ValueError("Invalid order for ABC condition")

        return coef

    def sommerfeld_element_matrices(self, idx):
        element = self.eqn.elements[idx]
        node_coords = self.eqn.nodes[element]

        S_e = np.zeros((9, 9), dtype=complex)
        gauss_points, gauss_weights = roots_legendre(3)

        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                # Shape functions and derivatives at this Gauss point
                N, dN_dxi, dN_deta = self.get_shape_functions(xi, eta)

                # Jacobian matrix
                J = np.zeros((2, 2))
                for k in range(9):
                    J[0, 0] += dN_dxi[k] * node_coords[k, 0]
                    J[0, 1] += dN_dxi[k] * node_coords[k, 1]
                    J[1, 0] += dN_deta[k] * node_coords[k, 0]
                    J[1, 1] += dN_deta[k] * node_coords[k, 1]

                # Determinant of Jacobian
                detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

                # Inverse of Jacobian_inner
                Jinv = (
                    np.array(
                        [
                            [J[1, 1], -J[0, 1]],
                            [-J[1, 0], J[0, 0]],
                        ]
                    )
                    / detJ
                )

                # Derivatives of shape functions with respect to x and y
                dN_dx = np.zeros(9)
                dN_dy = np.zeros(9)

                for k in range(9):
                    dN_dx[k] = Jinv[0, 0] * dN_dxi[k] + Jinv[0, 1] * dN_deta[k]
                    dN_dy[k] = Jinv[1, 0] * dN_dxi[k] + Jinv[1, 1] * dN_deta[k]

                # Weight for this Gauss point
                weight = gauss_weights[i] * gauss_weights[j] * detJ
                coef = self.abc_condition(order=self.abc_order)

                for m in range(9):
                    for n in range(9):
                        if (
                            np.linalg.norm(node_coords[m]) == self.eqn.outer_radius
                            and np.linalg.norm(node_coords[n]) == self.eqn.outer_radius
                        ):
                            S_e[m, n] += weight * coef * N[m] * N[n]

        return S_e

    def neumann_element_matrices(self, idx):
        # Get element nodes and coordinates
        element = self.eqn.elements[idx]
        node_coords = self.eqn.nodes[element]
        x, y = node_coords[:, 0], node_coords[:, 1]

        N_e = np.zeros(9, dtype=complex)
        gauss_points, gauss_weights = roots_legendre(3)

        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                # Shape functions and derivatives at this Gauss point
                N, dN_dxi, dN_deta = self.get_shape_functions(xi, eta)

                # Jacobian matrix
                J = np.zeros((2, 2))
                for k in range(9):
                    J[0, 0] += dN_dxi[k] * node_coords[k, 0]
                    J[0, 1] += dN_dxi[k] * node_coords[k, 1]
                    J[1, 0] += dN_deta[k] * node_coords[k, 0]
                    J[1, 1] += dN_deta[k] * node_coords[k, 1]

                # Determinant of Jacobian
                detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

                # Inverse of Jacobian_inner
                Jinv = (
                    np.array(
                        [
                            [J[1, 1], -J[0, 1]],
                            [-J[1, 0], J[0, 0]],
                        ]
                    )
                    / detJ
                )

                # Derivatives of shape functions with respect to x and y
                dN_dx = np.zeros(9)
                dN_dy = np.zeros(9)

                for k in range(9):
                    dN_dx[k] = Jinv[0, 0] * dN_dxi[k] + Jinv[0, 1] * dN_deta[k]
                    dN_dy[k] = Jinv[1, 0] * dN_dxi[k] + Jinv[1, 1] * dN_deta[k]

                # Weight for this Gauss point
                weight = gauss_weights[i] * gauss_weights[j] * detJ

                for m in range(9):
                    if np.linalg.norm(node_coords[m]) == self.eqn.inner_radius:
                        N_e[m] += weight * N[m] * self.get_normal_derivative(x[m], y[m])
                    else:
                        continue
        return N_e

    def assemble(self) -> None:
        self.K = np.zeros((self.eqn.n_nodes, self.eqn.n_nodes), dtype=complex)
        self.F = np.zeros(self.eqn.n_nodes, dtype=complex)
        self.S = np.zeros((self.eqn.n_nodes, self.eqn.n_nodes), dtype=complex)
        self.N = np.zeros(self.eqn.n_nodes, dtype=complex)

        for idx in self.eqn.inner_boundary_element_indices:
            N_e = self.neumann_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(9):
                self.N[global_indices[i]] += N_e[i]

        for idx in self.eqn.outer_boundary_element_indices:
            S_e = self.sommerfeld_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(9):
                for j in range(9):
                    self.S[global_indices[i], global_indices[j]] += S_e[i, j]

        for idx in range(self.eqn.n_elements):
            K_e, F_e = self.get_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(9):
                for j in range(9):
                    self.K[global_indices[i], global_indices[j]] += K_e[i, j]

                self.F[global_indices[i]] += F_e[i]

        self.K = self.K + self.S
        self.F = self.F + self.N

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        self.assemble()

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

        # Evaluate Hankel functions at unit radius and at r
        for m in range(-self.n_fourier, self.n_fourier + 1):
            # Hankel function at r
            h_r = hankel1(m, k * r)

            # Derivatives
            jp_m = jvp(m, k * r)
            hp_m = h1vp(m, k * r)

            # Contribution for positive n
            u_ex += (
                (self.eqn.i**m) * h_r * (jp_m / hp_m) * np.exp(self.eqn.i * m * theta)
            )

        return u_ex
