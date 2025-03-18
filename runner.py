from src.helmholtz import HelmHoltz
from src.solvers import FEM2DSolver
from src.viz import plot_comparison
import matplotlib.pyplot as plt

eqn = HelmHoltz()
solver = FEM2DSolver(eqn)
u_real, u_imag = solver.solve()

fig, ax = plot_comparison(solver, u_real, u_imag)
fig.savefig("assets/soln.png", dpi=300, bbox_inches="tight")
plt.close(fig)
