from src.helmholtz import HelmHoltz
from src.solvers import FEM2DSolver
from src.viz import plot_comparison, plot_mesh
import matplotlib.pyplot as plt

eqn = HelmHoltz()
solver = FEM2DSolver(eqn)
u_real, u_imag = solver.solve()

fig = plot_mesh(eqn)
fig.savefig("assets/mesh.png", dpi=300, bbox_inches="tight")

fig = plot_comparison(solver, u_real, u_imag)
fig.savefig("assets/soln.png", dpi=300, bbox_inches="tight")
plt.close(fig)
