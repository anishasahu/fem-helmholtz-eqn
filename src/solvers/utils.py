import numpy as np
from scipy.special import j0, y0, j1, y1, jvp, yvp, jv, yv


def get_solution(
    x, y, order, k_squared, inner_radius, outer_radius, n_fourier
) -> complex:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    k = np.sqrt(k_squared)
    a = inner_radius
    b = outer_radius

    u_ex = 0.0 + 0.0j
    for n in range(-n_fourier, n_fourier + 1):
        if order == 1:
            rhs = -(1j**n) * jvp(n, k * a)

            # Matrix for the linear system
            M = np.array(
                [
                    [jvp(n, k * a), yvp(n, k * a)],
                    [
                        jvp(n, k * b) - 1j * jv(n, k * b),
                        yvp(n, k * b) - 1j * yv(n, k * b),
                    ],
                ]
            )

            coeffs = np.linalg.solve(M, [rhs, 0])

            An = coeffs[0]
            Bn = coeffs[1]

            term = (An * jv(n, k * r) + Bn * yv(n, k * r)) * np.exp(1j * n * theta)
            u_ex += term
        elif order == 2:
            rhs = -(1j**n) * jvp(n, k * a)

            # Matrix for the linear system
            M = np.array(
                [
                    [jvp(n, k * a), yvp(n, k * a)],
                    [
                        k * jvp(n, k * b) - (1j * k - 1 / (2 * b)) * jv(n, k * b),
                        k * yvp(n, k * b) - (1j * k - 1 / (2 * b)) * yv(n, k * b),
                    ],
                ]
            )

            coeffs = np.linalg.solve(M, [rhs, 0])

            An = coeffs[0]
            Bn = coeffs[1]

            term = (An * jv(n, k * r) + Bn * yv(n, k * r)) * np.exp(1j * n * theta)
            u_ex += term
        elif order == 3:
            rhs = -(1j**n) * jvp(n, k * a)

            # Matrix for the linear system
            M = np.array(
                [
                    [jvp(n, k * a), yvp(n, k * a)],
                    [
                        (1j * k - 1 / b) * jvp(n, k * b)
                        + (0.5 * (2 * k**2 + 3j * k / b - 3 / (4 * b**2) + 1 / b**2))
                        * jv(n, k * b),
                        (1j * k - 1 / b) * yvp(n, k * b)
                        + (0.5 * (2 * k**2 + 3j * k / b - 3 / (4 * b**2) + 1 / b**2))
                        * yv(n, k * b),
                    ],
                ]
            )

            coeffs = np.linalg.solve(M, [rhs, 0])

            An = coeffs[0]
            Bn = coeffs[1]

            term = (An * jv(n, k * r) + Bn * yv(n, k * r)) * np.exp(1j * n * theta)
            u_ex += term
    return u_ex


def get_analytical_solution_sommerfeld(
    x, y, order, k_squared, inner_radius, outer_radius
):
    r = np.sqrt(x**2 + y**2)
    k = np.sqrt(k_squared)

    # Construct the coefficient matrix using Bessel functions
    if order == 1:
        matrix = np.array(
            [
                [
                    j0(k),
                    y0(k),
                ],  # Inner boundary condition
                [
                    jvp(0, k * outer_radius) - 1j * j0(k * outer_radius),
                    yvp(0, k * outer_radius) - 1j * y0(k * outer_radius),
                ],  # Outer boundary condition
            ]
        )

    elif order == 2:
        # ∂u/∂r - (ik - 1/(2R)) * u = 0 at r = R
        matrix = np.array(
            [
                [
                    j0(k),
                    y0(k),
                ],  # Inner boundary condition
                [
                    -j1(k * outer_radius) * k
                    - (1j * k + 1 / (2 * outer_radius)) * j0(k * outer_radius),
                    -y1(k * outer_radius) * k
                    - (1j * k + 1 / (2 * outer_radius)) * y0(k * outer_radius),
                ],  # Outer boundary condition
            ]
        )

    elif order == 3:
        matrix = np.array(
            [
                [
                    j0(k * inner_radius),
                    y0(k * inner_radius),
                ],  # Inner boundary condition
                [
                    -k * j1(k * outer_radius)
                    - 1j * k * j0(k * outer_radius)
                    - (1 / (2 * outer_radius)) * j0(k * outer_radius)
                    - (1j / (8 * k * outer_radius**2)) * j0(k * outer_radius),
                    -k * y1(k * outer_radius)
                    - 1j * k * y0(k * outer_radius)
                    - (1 / (2 * outer_radius)) * y0(k * outer_radius)
                    - (1j / (8 * k * outer_radius**2)) * y0(k * outer_radius),
                ],  # Outer boundary condition
            ]
        )

    # Right-hand side vector for boundary conditions
    rhs = np.array([1, 0])

    # Solve the system of equations to find coefficients A and B
    A, B = np.linalg.solve(matrix, rhs)

    # Compute and return the analytical solution at all node positions
    return A * j0(k * r) + B * y0(k * r)
