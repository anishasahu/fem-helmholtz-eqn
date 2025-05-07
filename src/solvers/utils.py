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

        rhs = -(1j**n) * jvp(n, k * a)

        if order == 0:
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
            
        elif order == 1:
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

        elif order == 2:
            # Matrix for the linear system
            M = np.array(
                [
                    [jvp(n, k * a), yvp(n, k * a)],
                    [
                        k * jvp(n, k * b) - (1j * k - (1.0 / (2.0 * b)) + (1j / (8 * k * b**2))) * jv(n, k * b),
                        k * yvp(n, k * b) - (1j * k - (1.0 / (2.0 * b)) + (1j / (8 * k * b**2))) * yv(n, k * b),
                    ],
                ]
            )

        elif order == 3:
            # Matrix for the linear system
            M = np.array(
                [
                    [jvp(n, k * a), yvp(n, k * a)],
                    [
                        k * jvp(n, k * b) - (1j * k - (1.0 / (2.0 * b)) - (1 / (8 * k * b**2)) + (1j / (8 * k**2 * b**3))) * jv(n, k * b),
                        k * yvp(n, k * b) - (1j * k - (1.0 / (2.0 * b)) - (1 / (8 * k * b**2)) + (1j / (8 * k**2 * b**3))) * yv(n, k * b),
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
    if order == 0:
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

    elif order == 1:
        # ∂u/∂r - (ik - 1/(2R)) * u = 0 at r = R
        matrix = np.array(
            [
                [
                    j0(k),
                    y0(k),
                ],  # Inner boundary condition
                [
                    k * jvp(0, k * outer_radius) - (1j * k - 1 / (2 * outer_radius)) * j0(k * outer_radius),
                    k * yvp(0, k * outer_radius) - (1j * k - 1 / (2 * outer_radius)) * y0(k * outer_radius),
                ],  # Outer boundary condition
            ]
        )

    elif order == 2:
        # ∂u/∂r - (ik - 1/(2R) + i/(8kR^2)) * u = 0 at r = R
        matrix = np.array(
            [
                [
                    j0(k), 
                    y0(k)
                ],
                [
                    k * jvp(0, k * outer_radius) - (1j * k - (1.0 / (2.0 * outer_radius)) + (1j / (8 * k * outer_radius**2))) * j0(k * outer_radius),
                    k * yvp(0, k * outer_radius) - (1j * k - (1.0 / (2.0 * outer_radius)) + (1j / (8 * k * outer_radius**2))) * y0(k * outer_radius)
                ],
            ]
        
        )
    elif order == 4:
        # ∂u/∂r - (ik - 1/(2R) - 1/(8kR^2) + i/(8k^2R^3)) * u = 0 at r = R
        matrix = np.array(
            [
                [
                    j0(k), 
                    y0(k)
                ],
                [
                    k * jvp(0, k * outer_radius) - (1j * k - (1.0 / (2.0 * outer_radius)) - (1.0 / (8.0 * k * outer_radius**2)) + (1j / (8 * k**2 * outer_radius**3))) * j0(k * outer_radius),
                    k * yvp(0, k * outer_radius) - (1j * k - (1.0 / (2.0 * outer_radius)) - (1.0 / (8.0 * k * outer_radius**2)) + (1j / (8 * k**2 * outer_radius**3))) * y0(k * outer_radius)
                ],
            ]
        )


    # Right-hand side vector for boundary conditions
    rhs = np.array([1, 0])

    # Solve the system of equations to find coefficients A and B
    A, B = np.linalg.solve(matrix, rhs)

    # Compute and return the analytical solution at all node positions
    return A * j0(k * r) + B * y0(k * r)
