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
        abc_order: int = 1,
    ):
        self.eqn = eqn
        self.k_squared = k_squared
        self.n_fourier = n_fourier
        self.abc_order = abc_order

    def get_shape_functions(self, xi, eta):
        # Shape functions for 9-node element
        N = np.zeros(4)
        dN_dxi = np.zeros(4)
        dN_deta = np.zeros(4)

        # Corner nodes
        N[0] = 0.25 * (1 - xi) * (1 - eta)
        N[1] = 0.25 * (1 + xi) * (1 - eta)
        N[2] = 0.25 * (1 + xi) * (1 + eta)
        N[3] = 0.25 * (1 - xi) * (1 + eta)

        # Derivatives of shape functions
        # Corner nodes
        dN_dxi[0] = -0.25 * (1 - eta)
        dN_dxi[1] = 0.25 * (1 - eta)
        dN_dxi[2] = 0.25 * (1 + eta)
        dN_dxi[3] = -0.25 * (1 + eta)

        dN_deta[0] = -0.25 * (1 - xi)
        dN_deta[1] = -0.25 * (1 + xi)
        dN_deta[2] = 0.25 * (1 + xi)
        dN_deta[3] = 0.25 * (1 - xi)

        return N, dN_dxi, dN_deta

    def get_element_matrices(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        element_nodes = self.eqn.elements[idx]
        node_coords = self.eqn.nodes[element_nodes]

        K_e = np.zeros((4, 4), dtype=complex)
        F_e = np.zeros(4, dtype=complex)

        gauss_points, gauss_weights = roots_legendre(4)

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
                        # Stiffness matrix: (∇φm·∇φn - k²φmφn)
                        K_e[m, n] += weight * detJ * (
                            -(dN_dx[m] * dN_dx[n] + dN_dy[m] * dN_dy[n])
                            + self.k_squared * N[m] * N[n]
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
            normal_derivative += (self.eqn.i**m) * k * jnp * np.exp(self.eqn.i * m * theta)

        return normal_derivative

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
            raise ValueError("Invalid order for ABC condition. Enter 1, 2 or 3.")

        return coef

    def sommerfeld_element_matrices(self, idx):
        element = self.eqn.elements[idx]
        node_coords = self.eqn.nodes[element]

        S_e = np.zeros((4, 4), dtype=complex)
        gauss_points, gauss_weights = roots_legendre(4)

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

            # This edge is on the outer boundary, perform line integration
            for i, s in enumerate(gauss_points):
                # Map the 1D Gauss point to the correct position on this edge
                if edge == (0, 1):  # Bottom edge
                    # xi varies from -1 to 1, eta = -1
                    xi = s
                    eta = -1
                elif edge == (1, 2):  # Right edge
                    # xi = 1, eta varies from -1 to 1
                    xi = 1
                    eta = s
                elif edge == (2, 3):  # Top edge
                    # xi varies from 1 to -1, eta = 1
                    xi = s
                    eta = 1
                elif edge == (3, 0):  # Left edge
                    # xi = -1, eta varies from 1 to -1
                    xi = -1
                    eta = s

                # Calculate shape functions at this point
                N, dN_dxi, dN_deta = self.get_shape_functions(xi, eta)

                # Jacobian matrix
                J = np.zeros((2, 2))
                for m in range(4):
                    J[0, 0] += dN_dxi[m] * node_coords[m, 0]
                    J[0, 1] += dN_dxi[m] * node_coords[m, 1]
                    J[1, 0] += dN_deta[m] * node_coords[m, 0]
                    J[1, 1] += dN_deta[m] * node_coords[m, 1]

                # Calculate differential length along the edge
                if edge == (0, 1) or edge == (2, 3):  # Horizontal edges
                    ds = np.sqrt(J[0, 0] ** 2 + J[0, 1] ** 2) #√[(∂x/∂ξ)² + (∂y/∂ξ)²]
                else:  # Vertical edges
                    ds = np.sqrt(J[1, 0] ** 2 + J[1, 1] ** 2) #√[(∂x/∂η)² + (∂y/∂η)²]

                # Sommerfeld coefficient (typically i*k for 2D Helmholtz)
                sommerfeld_coef = self.abc_condition(self.abc_order)

                # Weight for this Gauss point
                weight = gauss_weights[i]

                # Compute the Sommerfeld matrix for this point
                for m in range(4):
                    for n in range(4):
                        S_e[m, n] += N[m] * N[n] * sommerfeld_coef * ds * weight


        return S_e

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
        gauss_points, gauss_weights = roots_legendre(4)

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

            # This edge is on the inner boundary, perform line integration
            for i, s in enumerate(gauss_points):
                # Parameter s goes from -1 to 1 along the edge
                # Map the Gauss point to the correct position on this edge
                if edge == (0, 1):  # Bottom edge
                    # xi varies from -1 to 1, eta = -1
                    xi = s
                    eta = -1
                elif edge == (1, 2):  # Right edge
                    # xi = 1, eta varies from -1 to 1
                    xi = 1
                    eta = s
                elif edge == (2, 3):  # Top edge
                    # xi varies from 1 to -1, eta = 1
                    xi = s
                    eta = 1
                elif edge == (3, 0):  # Left edge
                    # xi = -1, eta varies from 1 to -1
                    xi = -1
                    eta = s

                # Calculate shape functions and derivatives at this point
                N, dN_dxi, dN_deta = self.get_shape_functions(xi, eta)

                # Jacobian matrix
                J = np.zeros((2, 2))
                for k in range(4):
                    J[0, 0] += dN_dxi[k] * node_coords[k, 0]
                    J[0, 1] += dN_dxi[k] * node_coords[k, 1]
                    J[1, 0] += dN_deta[k] * node_coords[k, 0]
                    J[1, 1] += dN_deta[k] * node_coords[k, 1]

                # Calculate differential length along the edge
                if edge == (0, 1) or edge == (2, 3):  # Horizontal edges
                    ds = np.sqrt(J[0, 0] ** 2 + J[0, 1] ** 2) #√[(∂x/∂ξ)² + (∂y/∂ξ)²]
                else:  # Vertical edges
                    ds = np.sqrt(J[1, 0] ** 2 + J[1, 1] ** 2) #√[(∂x/∂η)² + (∂y/∂η)²]

                # Current position
                current_pos = np.zeros(2)
                for k in range(4):
                    current_pos[0] += N[k] * node_coords[k, 0] 
                    current_pos[1] += N[k] * node_coords[k, 1] 

                x, y = current_pos
                
                # Get the normal derivative using the function
                normal_derivative = self.get_normal_derivative(x, y)

                # Weight for this Gauss point
                weight = gauss_weights[i]

                # Add contribution to element vector
                for m in range(4):
                    N_e[m] += N[m] * normal_derivative * ds * weight

        return N_e

    def assemble(self) -> None:
        self.K = np.zeros((self.eqn.n_nodes, self.eqn.n_nodes), dtype=complex)
        self.F = np.zeros(self.eqn.n_nodes, dtype=complex)
        self.S = np.zeros((self.eqn.n_nodes, self.eqn.n_nodes), dtype=complex)
        self.N = np.zeros(self.eqn.n_nodes, dtype=complex)

        for idx in self.eqn.inner_boundary_element_indices:
            N_e = self.neumann_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(4):
                self.N[global_indices[i]] += N_e[i]

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
