import matplotlib.pyplot as plt

from src.helmholtz import HelmHoltz
from src.utils import get_solver
from src.viz import plot_comparison, plot_mesh, plot_convergence

k_squared_values = [1, 10]
n_r_n_theta_values = [(5, 5), (9, 10)]

# eqn = HelmHoltz()
# fig = plot_mesh(eqn)
# fig.savefig("assets/mesh.png", dpi=300, bbox_inches="tight")
# plt.close(fig)

# types = ["dirichlet", "neumann_dirichlet", "dirichlet_sommerfeld", "default"]
types = ["default"]

for type in types:
    solver = get_solver(type, eqn)
    u_real, u_imag = solver.solve()
    fig = plot_comparison(solver, u_real, u_imag)
    fig.write_image(f"assets/{type}_soln.png")

for type in types:
    fig = plot_convergence(type, k_squared_values, n_r_n_theta_values)
    fig.savefig(f"assets/convergence_plot_{type}.png", dpi=300)
    plt.close(fig)
