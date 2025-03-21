import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_comparison(solver, u_real, u_imag, grid_points=50, fig_size=(12, 8)):
    theta = np.linspace(0, 2 * np.pi, grid_points)
    r = np.linspace(solver.eqn.inner_radius, solver.eqn.outer_radius, grid_points)
    
    # Create meshgrid for polar coordinates
    r_grid, theta_grid = np.meshgrid(r, theta)
    
    # Convert to Cartesian coordinates
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    
    # Calculate exact solution on the grid
    u_exact = np.zeros((grid_points, grid_points), dtype=complex)
    for i in range(grid_points):
        for j in range(grid_points):
            u_exact[i, j] = solver.get_analytical_solution(x_grid[i, j], y_grid[i, j])
    
    # Compute magnitude of exact solution
    map_exact = np.sqrt(np.real(u_exact) ** 2 + np.imag(u_exact) ** 2)
    
    # Create a single plot for comparison
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot numerical solution (green)
    for element in solver.eqn.elements:
        x = solver.eqn.nodes[element, 0]
        y = solver.eqn.nodes[element, 1]
        u_mag = np.sqrt(u_real**2 + u_imag**2)
        z = u_mag[element]
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts, alpha=0.6, color="green", label="Numerical")
        ax.add_collection3d(poly)
    
    # Create a custom legend
    numerical_patch = plt.Rectangle((0, 0), 1, 1, color="green", alpha=0.6)
    analytical_patch = plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.5)
    ax.legend([numerical_patch, analytical_patch], ["Numerical Solution", "Analytical Solution"], 
              loc="upper right")
    
    # Set labels and title
    ax.set_title("Comparison of Numerical and Analytical Solutions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("|u|")
    
    plt.tight_layout()
    
    return fig


def plot_mesh(eqn):
    """Plot the generated mesh with nodes and elements."""
    fig, ax = plt.subplots()

    # Plot all nodes
    ax.scatter(
        eqn.nodes[:, 0], eqn.nodes[:, 1], color="blue", s=1, zorder=3, label="Nodes"
    )

    # Highlight boundary nodes
    ax.scatter(
        eqn.inner_boundary_nodes[:, 0],
        eqn.inner_boundary_nodes[:, 1],
        color="red",
        s=1,
        zorder=4,
        label="Inner Boundary",
    )
    ax.scatter(
        eqn.outer_boundary_nodes[:, 0],
        eqn.outer_boundary_nodes[:, 1],
        color="green",
        s=1,
        zorder=4,
        label="Outer Boundary",
    )

    # Plot element connectivity (edges)
    for element in eqn.elements:
        element_coords = eqn.nodes[element]
        for i in range(4):  # Loop through the 4 corners of each quadrilateral
            x_coords = [element_coords[i, 0], element_coords[(i + 1) % 4, 0]]
            y_coords = [element_coords[i, 1], element_coords[(i + 1) % 4, 1]]
            ax.plot(x_coords, y_coords, color="black", lw=1.5, zorder=2)

    # Set plot attributes
    ax.set_aspect("equal")
    ax.set_title("Mesh Plot with Nodes and Elements")
    ax.legend(loc="upper right")

    plt.grid(True)
    plt.tight_layout()

    return fig
