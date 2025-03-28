from src.helmholtz import HelmHoltz
from src.solvers import (
    FEM2DDirichletSolver,
    FEM2DDirichletSommerfeldSolver,
    FEM2DNeumannDirichletSolver,
    FEM2DSolver,
)
from src.viz import plot_comparison, plot_mesh

eqn = HelmHoltz()
solver0 = FEM2DDirichletSolver(eqn)
u_real0, u_imag0 = solver0.solve()

solver1 = FEM2DNeumannDirichletSolver(eqn)
u_real1, u_imag1 = solver1.solve()

solver2 = FEM2DDirichletSommerfeldSolver(eqn)
u_real2, u_imag2 = solver2.solve()

solver3 = FEM2DSolver(eqn)
u_real3, u_imag3 = solver3.solve()

solver3 = FEM2DSolver(eqn)
u_real3, u_imag3 = solver3.solve()

solver3 = FEM2DSolver(eqn)
u_real3, u_imag3 = solver3.solve()

fig = plot_mesh(eqn)
fig.savefig("assets/mesh.png", dpi=300, bbox_inches="tight")

fig = plot_comparison(solver0, u_real0, u_imag0, tag=["dirichlet"])
fig.write_image("assets/dirichlet_soln.png")

fig = plot_comparison(solver1, u_real1, u_imag1, tag=["neumann_dirichlet"])
fig.write_image("assets/neumann_dirichlet_sol.png")

fig = plot_comparison(solver2, u_real2, u_imag2, tag=["dirichlet_sommerfeld"])
fig.write_image("assets/dirichlet_sommerfeld_sol.png")
