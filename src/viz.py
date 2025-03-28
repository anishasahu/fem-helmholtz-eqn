import time

import wandb
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go
from dotenv import load_dotenv


def plot_comparison(solver, u_real, u_imag, tag, grid_points=50, fig_size=(800, 600)):
    theta = np.linspace(0, 2 * np.pi, grid_points)
    r = np.linspace(solver.eqn.inner_radius, solver.eqn.outer_radius, grid_points)

    # Create meshgrid for polar coordinates
    r_grid, theta_grid = np.meshgrid(r, theta)

    # Convert to Cartesian coordinates
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    # Calculate exact solution on the grid
    u_exact = np.zeros((grid_points, grid_points), dtype=complex)
    start = time.perf_counter()
    for i in range(grid_points):
        for j in range(grid_points):
            u_exact[i, j] = solver.get_analytical_solution(x_grid[i, j], y_grid[i, j])
    end = time.perf_counter()
    print(f"Time {end - start:.4f} seconds")

    # Compute magnitude of exact solution
    map_exact = np.sqrt(np.real(u_exact) ** 2 + np.imag(u_exact) ** 2)

    # Plotly plot for comparison
    fig = go.Figure()

    # Plot numerical solution (green mesh)
    for element in solver.eqn.elements:
        x = solver.eqn.nodes[element, 0]
        y = solver.eqn.nodes[element, 1]
        u_mag = np.sqrt(u_real**2 + u_imag**2)
        z = u_mag[element]
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color="green",
                opacity=0.6,
                name="Numerical Solution",
                alphahull=0,
            )
        )

    # Plot exact solution (red surface)
    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=map_exact,
            colorscale="reds",
            opacity=0.5,
            name="Analytical Solution",
        )
    )

    # Update layout for interactivity
    fig.update_layout(
        title="Comparison of Numerical and Analytical Solutions",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="|u|"),
        width=fig_size[0],
        height=fig_size[1],
        legend=dict(x=0.8, y=0.95),
    )

    fig.show()

    # upload fig to wandb
    load_dotenv()
    wandb.init(project="fem-helmholtz", entity="sauravmaheshkar", tags=tag)
    wandb.log(
        {
            "Numerical Solution (Green) vs. Analyitcal Solution (Red) with Dirichlet Conditions": fig
        }
    )
    wandb.finish()

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
