import gmsh
import numpy as np


class HelmHoltz:
    def __init__(
        self,
        inner_radius: float = 1.0,
        outer_radius: float = 1.5,
        n_theta: int = 10,
        n_r: int = 9,
        abc_order: int = 1,
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
        gmsh.initialize()
        gmsh.model.add("annulus")

        # Define the inner and outer circular boundaries
        outer = gmsh.model.occ.addCircle(0, 0, 0, self.outer_radius)
        inner = gmsh.model.occ.addCircle(0, 0, 0, self.inner_radius)

        # Create curve loops
        outer_loop = gmsh.model.occ.addCurveLoop([outer])
        inner_loop = gmsh.model.occ.addCurveLoop([inner])

        # Create surface with inner loop as a hole
        surface = gmsh.model.occ.addPlaneSurface([outer_loop, inner_loop])  # noqa: F841

        gmsh.model.occ.synchronize()

        # Define mesh size dynamically to prevent errors in large radius ranges
        print(f"n_r: {self.n_r}, n_theta: {self.n_theta}")
        self.mesh_size = min(
            (self.outer_radius - self.inner_radius) / self.n_r,
            (2 * np.pi * self.inner_radius) / self.n_theta,  # Ensures inner mesh is not too coarse
            (2 * np.pi * self.outer_radius) / self.n_theta
        )
        print(f"Mesh size: {self.mesh_size}")

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.mesh_size)

        # Generate 2D mesh with quadrilateral elements
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()
        gmsh.model.mesh.setOrder(1)  # Bilinear elements

        # Extract node coordinates
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        self.nodes = np.array(node_coords).reshape(-1, 3)[:, :2]
        self.n_nodes = len(self.nodes)

        # Extract element connectivity (handle different element types)
        self.elements = []

        elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(2)

        for i, etype in enumerate(elem_types):
            num_nodes_per_element = len(node_tags[i]) // len(elem_tags[i])  # Dynamically determine size
            self.elements.append(np.array(node_tags[i]).reshape(-1, num_nodes_per_element) - 1)

        self.elements = np.vstack(self.elements) if self.elements else np.array([])

        self.n_elements = len(self.elements)

        # Identify boundary nodes
        self.inner_boundary_nodes = []
        self.outer_boundary_nodes = []

        # Identify boundary nodes by their indices rather than coordinates
        self.inner_boundary_node_indices = []
        self.outer_boundary_node_indices = []

        # Calculate distances of all nodes from origin
        node_distances = np.linalg.norm(self.nodes, axis=1)

        # Find indices of nodes on inner boundary
        self.inner_boundary_node_indices = np.where(
            np.isclose(node_distances, self.inner_radius, atol=1e-5)
        )[0]

        # Find indices of nodes on outer boundary
        self.outer_boundary_node_indices = np.where(
            np.isclose(node_distances, self.outer_radius, atol=1e-5)
        )[0]

        # Store boundary nodes
        self.inner_boundary_nodes = self.nodes[self.inner_boundary_node_indices]
        self.outer_boundary_nodes = self.nodes[self.outer_boundary_node_indices]

        self.inner_boundary_element_indices = []
        self.outer_boundary_element_indices = []

        for idx, element in enumerate(self.elements):
            if any(node in element for node in self.inner_boundary_node_indices):
                self.inner_boundary_element_indices.append(idx)
            if any(node in element for node in self.outer_boundary_node_indices):
                self.outer_boundary_element_indices.append(idx)

        gmsh.finalize()

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
