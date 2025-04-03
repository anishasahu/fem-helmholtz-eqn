from src.solvers.fem2d import FEM2DSolver
from src.solvers.fem2d_dirichlet import FEM2DDirichletSolver
from src.solvers.fem2d_neumann_dirichlet import FEM2DNeumannDirichletSolver
from src.solvers.fem2d_dirichlet_sommerfeld import FEM2DDirichletSommerfeldSolver


solvers = ["default", "dirichlet", "dirichlet_sommerfeld", "neumann_dirichlet"]


def get_solver(type: str, eqn, **kwargs):
    assert type in solvers, f"Invalid solver type: {type} expected one of: {solvers}"

    if type == "default":
        return FEM2DSolver(eqn, **kwargs)
    elif type == "dirichlet":
        return FEM2DDirichletSolver(eqn, **kwargs)
    elif type == "neumann_dirichlet":
        return FEM2DNeumannDirichletSolver(eqn, **kwargs)
    elif type == "dirichlet_sommerfeld":
        return FEM2DDirichletSommerfeldSolver(eqn, **kwargs)

    raise AssertionError(f"Unhandled solver type: {type}")
