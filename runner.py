import matplotlib.pyplot as plt

from src.helmholtz import HelmHoltz
from src.utils import get_solver
from src.viz import plot_comparison, plot_mesh, plot_convergence

k_squared_values = [1, 10]
n_r_values = [5, 7, 9]

eqn = HelmHoltz()
fig = plot_mesh(eqn)
fig.savefig("assets/mesh.png", dpi=300, bbox_inches="tight")
plt.close(fig)

for type in ["dirichlet", "neumann_dirichlet", "dirichlet_sommerfeld", "default"]:
    solver = get_solver(type, eqn)
    u_real, u_imag = solver.solve()
    fig = plot_comparison(solver, u_real, u_imag)
    fig.write_image(f"assets/{type}_soln.png")

fig = plot_convergence("dirichlet", k_squared_values, n_r_values)
fig.savefig("assets/convergence_plot.png", dpi=300)
plt.close(fig)
