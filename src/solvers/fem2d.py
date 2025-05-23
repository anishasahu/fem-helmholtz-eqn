from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import h1vp, hankel1, jvp, roots_legendre

from .base import BaseSolver
from .utils import get_solution


class FEM2DSolver(BaseSolver):
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
                        global_n = edge[local_n]
                        S_e[global_m, global_n] += (
                            N[local_m] * N[local_n] * sommerfeld_coef * ds * weight
                        )

        return S_e

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

    def assemble(self) -> None:
        self.K = np.zeros((self.eqn.n_nodes, self.eqn.n_nodes), dtype=complex)
        self.F = np.zeros(self.eqn.n_nodes, dtype=complex)
        self.S = np.zeros((self.eqn.n_nodes, self.eqn.n_nodes), dtype=complex)
        self.N_global = np.zeros(self.eqn.n_nodes, dtype=complex)

        for idx in self.eqn.inner_boundary_element_indices:
            N_e = self.neumann_element_matrices(idx)
            global_indices = self.eqn.elements[idx]

            for i in range(4):
                self.N_global[global_indices[i]] += N_e[i]

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
        self.F += self.N_global

    def solve(self) -> Tuple[NDArray, NDArray]:
        self.assemble()

        u_complex = np.linalg.solve(self.K, self.F)

        u_real = np.real(u_complex)
        u_imag = np.imag(u_complex)

        return u_real, u_imag

    def get_analytical_solution(self, x, y) -> complex:
        # Initialize the analytical solution with zero (complex number)
        u_ex = 0.0 + 0.0j

        # Convert Cartesian coordinates (x,y) to polar coordinates (r,θ)
        r = np.sqrt(x**2 + y**2)  # Radial distance from origin
        theta = np.arctan2(y, x)  # Angular position (in radians)

        # Calculate wavenumber from the square of the wavenumber property
        k = np.sqrt(self.k_squared)

        # Get the outer radius parameter (boundary of computational domain)
        a = self.outer_radius

        ans = get_solution(
            x,
            y,
            self.abc_order,
            self.k_squared,
            self.inner_radius,
            self.outer_radius,
            self.n_fourier,
        )

        # Sum over Fourier modes from -n_fourier to n_fourier
        for n in range(-self.n_fourier, self.n_fourier + 1):
            # Calculate derivative of Bessel function of first kind at outer boundary
            j_prime = jvp(n, k * a)  # J_n'(ka)

            # Calculate derivative of Hankel function of first kind at outer boundary
            h1_prime = h1vp(n, k * a)  # H_n^(1)'(ka)

            # Calculate Hankel function of first kind at the evaluation point
            h1 = hankel1(n, k * r)  # H_n^(1)(kr)

            # Add contribution of this mode to the solution
            # Formula implements the analytical solution for scattered wave:
            # u(r,θ) = -∑ i^n·e^(inθ)·[J_n'(ka)/H_n^(1)'(ka)]·H_n^(1)(kr)
            u_ex -= 1j**n * np.exp(1j * n * theta) * j_prime / h1_prime * h1

        # Return the complete analytical solution at point (x,y)
        return ans
