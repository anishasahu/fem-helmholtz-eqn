import gmsh
import numpy as np


class HelmHoltz:
    def __init__(
        self,
        inner_radius=1.0,
        outer_radius=1.5,
        n_theta=10,
        n_r=9,
        abc_order=1,
    ):
        # Global parameters
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.abc_order = abc_order

        # Mesh parameters
        self.n_theta = n_theta
        self.n_r = n_r

        self.n_theta_nodes = 2 * self.n_theta + 1
        self.n_r_nodes = 2 * self.n_r + 1

        # Complex number i
        self.i = complex(0.0, 1.0)

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

        # Define mesh size
        mesh_size = (self.outer_radius - self.inner_radius) / self.n_r

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

        # Generate 2D mesh with bilinear quadrilateral elements
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()
        gmsh.model.mesh.setOrder(1)  # Changed to bilinear elements (order 1)

        # Extract node coordinates
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        self.nodes = np.array(node_coords).reshape(-1, 3)[:, :2]
        self.n_nodes = len(self.nodes)

        # Extract element connectivity
        self.elements = []

        elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(2)

        for i, etype in enumerate(elem_types):
            self.elements = (
                np.array(node_tags[i]).reshape(-1, 4) - 1
            )  # Changed from 9 to 4 nodes per element

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

        # Also keep the actual coordinates for convenience if needed
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
