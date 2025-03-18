import numpy as np


class HelmHoltz:
    def __init__(
        self,
        inner_radius=1.0,
        outer_radius=1.5,
        n_theta=15,
        n_r=5,
        use_dtn_bc=True,
        abc_order=3,
    ):
        # Global parameters
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.use_dtn_bc = use_dtn_bc
        self.abc_order = abc_order

        # Mesh parameters
        self.n_theta = n_theta
        self.n_r = n_r

        # Complex number i
        self.i = complex(0.0, 1.0)

        # Initialize mesh and solution vectors
        self.generate_mesh()
        self.u_real = np.zeros(self.n_nodes)
        self.u_imag = np.zeros(self.n_nodes)

    def generate_mesh(self):
        """Generate annular mesh with quadratic elements."""
        # Number of nodes in each direction (quadratic elements)
        self.n_theta_nodes = 2 * self.n_theta + 1
        self.n_r_nodes = 2 * self.n_r + 1

        # Total number of nodes
        self.n_nodes = self.n_theta_nodes * self.n_r_nodes

        # Create node coordinates
        self.nodes = np.zeros((self.n_nodes, 2))

        # Angle and radius increments
        d_theta = 2 * np.pi / self.n_theta
        d_r = (self.outer_radius - self.inner_radius) / self.n_r

        # Create nodes
        node_idx = 0
        for r_idx in range(self.n_r_nodes):
            r = self.inner_radius + (r_idx * d_r / 2)
            for theta_idx in range(self.n_theta_nodes):
                theta = (theta_idx * d_theta / 2) % (2 * np.pi)

                # Convert to Cartesian coordinates
                x = r * np.cos(theta)
                y = r * np.sin(theta)

                self.nodes[node_idx] = [x, y]
                node_idx += 1

        # Create element connectivity for quadratic elements
        self.elements = []

        for r_idx in range(self.n_r):
            for theta_idx in range(self.n_theta):
                # Calculate node indices for this element (9 nodes for quadratic element)
                element_nodes = []

                # Corner nodes
                r1_t1 = (2 * r_idx) * self.n_theta_nodes + (2 * theta_idx)
                r1_t2 = (2 * r_idx) * self.n_theta_nodes + (
                    (2 * theta_idx + 2) % self.n_theta_nodes
                )
                r2_t1 = (2 * r_idx + 2) * self.n_theta_nodes + (2 * theta_idx)
                r2_t2 = (2 * r_idx + 2) * self.n_theta_nodes + (
                    (2 * theta_idx + 2) % self.n_theta_nodes
                )

                # Mid-side nodes
                r1_tmid = (2 * r_idx) * self.n_theta_nodes + (
                    (2 * theta_idx + 1) % self.n_theta_nodes
                )
                r2_tmid = (2 * r_idx + 2) * self.n_theta_nodes + (
                    (2 * theta_idx + 1) % self.n_theta_nodes
                )
                rmid_t1 = (2 * r_idx + 1) * self.n_theta_nodes + (2 * theta_idx)
                rmid_t2 = (2 * r_idx + 1) * self.n_theta_nodes + (
                    (2 * theta_idx + 2) % self.n_theta_nodes
                )

                # Center node
                rmid_tmid = (2 * r_idx + 1) * self.n_theta_nodes + (
                    (2 * theta_idx + 1) % self.n_theta_nodes
                )

                # Add nodes in counter-clockwise order (for quadratic element)
                element_nodes = [
                    r1_t1,
                    r1_tmid,
                    r1_t2,
                    rmid_t2,
                    r2_t2,
                    r2_tmid,
                    r2_t1,
                    rmid_t1,
                    rmid_tmid,
                ]
                self.elements.append(element_nodes)

        # Store element count
        self.n_elements = len(self.elements)

        # Identify boundary nodes
        self.inner_boundary_nodes = []
        self.outer_boundary_nodes = []

        for i, node in enumerate(self.nodes):
            r = np.sqrt(node[0] ** 2 + node[1] ** 2)
            if abs(r - self.inner_radius) < 1e-10:
                self.inner_boundary_nodes.append(i)
            elif abs(r - self.outer_radius) < 1e-10:
                self.outer_boundary_nodes.append(i)
