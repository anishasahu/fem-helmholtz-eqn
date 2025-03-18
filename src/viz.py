import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(solver, u_real, u_imag, grid_points=50, fig_size=(18, 6)):
    # Create a regular grid for plotting the exact solution
    r_min = solver.eqn.inner_radius
    r_max = solver.eqn.outer_radius

    theta = np.linspace(0, 2 * np.pi, grid_points)
    r = np.linspace(r_min, r_max, grid_points)

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

    # Extract numerical solution and reshape to structured grid
    n_theta_nodes = solver.eqn.n_theta_nodes
    n_r_nodes = solver.eqn.n_r_nodes

    x_fem = np.zeros((n_r_nodes, n_theta_nodes))
    y_fem = np.zeros((n_r_nodes, n_theta_nodes))
    mag_fem = np.zeros((n_r_nodes, n_theta_nodes))

    for r_idx in range(n_r_nodes):
        for theta_idx in range(n_theta_nodes):
            node_idx = r_idx * n_theta_nodes + theta_idx
            x_fem[r_idx, theta_idx] = solver.eqn.nodes[node_idx, 0]
            y_fem[r_idx, theta_idx] = solver.eqn.nodes[node_idx, 1]
            mag_fem[r_idx, theta_idx] = np.sqrt(
                u_real[node_idx] ** 2 + u_imag[node_idx] ** 2
            )

    # Compute magnitude of exact solution
    map_exact = np.abs(u_exact)

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=fig_size, subplot_kw={"projection": "3d"})

    # Plot numerical solution
    axes[0].plot_surface(
        x_fem,
        y_fem,
        mag_fem,
        color="green",
        edgecolor="none",
        alpha=0.9,
    )
    axes[0].set_title("Numerical Solution")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_zlabel("|u|")

    # Plot exact solution
    axes[1].plot_surface(
        x_grid,
        y_grid,
        map_exact,
        color="red",
        edgecolor="none",
        alpha=0.9,
    )
    axes[1].set_title("Analytical Solution")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].set_zlabel("|u|")

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
