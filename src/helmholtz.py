import numpy as np


class HelmHoltz:
    def __init__(
        self,
        inner_radius: float = 1.0,
        outer_radius: float = 1.5,
        n_theta: int = 10,
        n_r: int = 10,
        abc_order: int = 3,
    ) -> None:
        # Global parameters
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.abc_order = abc_order

        # Mesh parameters
        self.n_theta = n_theta
        self.n_r = n_r

        self.n_theta_nodes = 2 * self.n_theta + 1
        self.n_r_nodes = 2 * self.n_r + 1

        # Initialize mesh and solution vectors
        self.generate_mesh()
        self.u_real = np.zeros(self.n_nodes)
        self.u_imag = np.zeros(self.n_nodes)

    def generate_mesh(self):
        # Create structured mesh in polar coordinates
        r = np.linspace(self.inner_radius, self.outer_radius, self.n_r_nodes)
        theta = np.linspace(0, 2 * np.pi, self.n_theta_nodes, endpoint=False)

        R, T = np.meshgrid(r, theta, indexing="ij")

        # Convert to Cartesian coordinates
        x = R * np.cos(T)
        y = R * np.sin(T)

        # Flatten and combine to get list of nodes
        self.nodes = np.column_stack((x.ravel(), y.ravel()))
        self.n_nodes = self.nodes.shape[0]

        # Generate structured quad elements
        self.elements = []
        for i in range(self.n_r_nodes - 1):
            for j in range(self.n_theta_nodes):
                n0 = i * self.n_theta_nodes + j
                n1 = i * self.n_theta_nodes + (j + 1) % self.n_theta_nodes
                n2 = (i + 1) * self.n_theta_nodes + (j + 1) % self.n_theta_nodes
                n3 = (i + 1) * self.n_theta_nodes + j
                self.elements.append([n0, n1, n2, n3])

        self.elements = np.array(self.elements)
        self.n_elements = self.elements.shape[0]

        # Identify boundary nodes
        node_distances = np.linalg.norm(self.nodes, axis=1)

        self.inner_boundary_node_indices = np.where(
            np.isclose(node_distances, self.inner_radius, atol=1e-5)
        )[0]
        self.outer_boundary_node_indices = np.where(
            np.isclose(node_distances, self.outer_radius, atol=1e-5)
        )[0]

        self.inner_boundary_nodes = self.nodes[self.inner_boundary_node_indices]
        self.outer_boundary_nodes = self.nodes[self.outer_boundary_node_indices]

        # Identify elements adjacent to inner/outer boundary nodes
        self.inner_boundary_element_indices = []
        self.outer_boundary_element_indices = []

        for idx, element in enumerate(self.elements):
            if any(node in element for node in self.inner_boundary_node_indices):
                self.inner_boundary_element_indices.append(idx)
            if any(node in element for node in self.outer_boundary_node_indices):
                self.outer_boundary_element_indices.append(idx)

    def get_abc_coeff(self, k_squared, order: int) -> complex:
        k = np.sqrt(k_squared)

        # set default value
        coef = 1j * k

        if order == 1:
            coef = 1j * k
        elif order == 2:
            coef = 1j * k + 1.0 / (2.0 * self.outer_radius)
        elif order == 3:
            for i, ith_node in enumerate(self.outer_boundary_node_indices):
                theta_i = np.arctan2(self.nodes[ith_node, 1], self.nodes[ith_node, 0])
                for j, jth_node in enumerate(self.outer_boundary_node_indices):
                    theta_j = np.arctan2(
                        self.nodes[jth_node, 1], self.nodes[jth_node, 0]
                    )
                    if abs(theta_i - theta_j) < 0.5:
                        coef = (
                            1j * k
                            + 1.0 / (2.0 * self.outer_radius)
                            + 1j / (8.0 * k * self.outer_radius**2)
                        )
        else:
            raise ValueError("Invalid order for ABC condition. Enter 1, 2 or 3.")

        return coef
