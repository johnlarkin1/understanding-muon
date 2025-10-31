"""
Shared utilities for Muon optimizer visualizations.
Contains mathematical functions, constants, and reusable drawing functions.
"""

import numpy as np
from manim import (
    BLUE_D,
    BLUE_E,
    GRAY_D,
    GREY_B,
    PURPLE_E,
    TEAL_E,
    ImplicitFunction,
    Surface,
    Tex,
    ThreeDAxes,
    VGroup,
    interpolate_color,
)

# =============================
# Problem setup (math utilities)
# =============================

# Domain and ranges
X_RANGE = (-5, 5, 1)
Y_RANGE = (-5, 5, 1)
# Over the [-5,5]^2 box, z H [-78.33, ~250]
Z_RANGE = (-100, 100, 50)

GLOBAL_MIN_COORD = -2.903534  # 1D optimum
GLOBAL_MIN_2D = np.array([GLOBAL_MIN_COORD, GLOBAL_MIN_COORD], dtype=float)

LOCAL_MINIMA_COORDS = [
    np.array([2.746803, -2.903534]),
    np.array([-2.903534, 2.746803]),
    np.array([2.746803, 2.746803]),
]

# =============================
# Camera params
# =============================
ZOOM = 0.55


def styblinski_tang_fn(x: float, y: float) -> float:
    """StyblinskiTang function in 2D."""
    return 0.5 * ((x**4 - 16 * x**2 + 5 * x) + (y**4 - 16 * y**2 + 5 * y))


def styblinski_tang_grad(x: float, y: float) -> np.ndarray:
    """Gradient of 2D StyblinskiTang."""
    dfx = 2 * x**3 - 16 * x + 2.5
    dfy = 2 * y**3 - 16 * y + 2.5
    return np.array([dfx, dfy], dtype=float)


def styblinski_tang_hessian(x: float, y: float) -> np.ndarray:
    """Hessian matrix of 2D StyblinskiTang.

    Returns 2x2 matrix of second derivatives:
    [[∂²f/∂x², ∂²f/∂x∂y],
     [∂²f/∂y∂x, ∂²f/∂y²]]
    """
    # For Styblinski-Tang, the function is separable: f(x,y) = 0.5[g(x) + g(y)]
    # where g(u) = u^4 - 16u^2 + 5u
    # So cross-derivatives are zero: ∂²f/∂x∂y = 0

    # ∂²f/∂x² = 0.5 * d²g/dx² = 0.5 * (12x² - 32) = 6x² - 16
    d2f_dx2 = 6 * x**2 - 16

    # ∂²f/∂y² = 0.5 * d²g/dy² = 0.5 * (12y² - 32) = 6y² - 16
    d2f_dy2 = 6 * y**2 - 16

    return np.array([[d2f_dx2, 0.0], [0.0, d2f_dy2]], dtype=float)


# ============================
# Reusable drawing/scene utils
# ============================


def make_3d_axes() -> tuple[ThreeDAxes, VGroup]:
    """Create 3D axes for the Styblinski-Tang function visualization."""
    axes = ThreeDAxes(
        x_range=X_RANGE,
        y_range=Y_RANGE,
        z_range=Z_RANGE,
        x_length=8,
        y_length=8,
        z_length=5,
        tips=False,
    )
    # Only label the z-axis (remove x and y labels to avoid confusion)
    z_label = Tex("z").scale(0.8)
    z_label.move_to(axes.c2p(0, 0, Z_RANGE[1] + 10))

    labels = VGroup(z_label)
    return axes, labels


def make_surface(axes: ThreeDAxes) -> Surface:
    """Create the 3D surface for the Styblinski-Tang function."""
    # Parametric surface z = f(x,y)
    surf = Surface(
        lambda u, v: axes.c2p(u, v, styblinski_tang_fn(u, v)),
        u_range=(X_RANGE[0], X_RANGE[1]),
        v_range=(Y_RANGE[0], Y_RANGE[1]),
        resolution=(64, 64),
        fill_opacity=0.9,
        checkerboard_colors=[  # pleasant dual-color shading
            interpolate_color(BLUE_E, PURPLE_E, 0.35),
            interpolate_color(TEAL_E, BLUE_D, 0.35),
        ],
    )
    surf.set_stroke(width=0.25, opacity=0.2)
    return surf


def make_contours(
    levels: list[float], x_range=X_RANGE, y_range=Y_RANGE, color=GREY_B
) -> VGroup:
    """
    Use ImplicitFunction to render contour lines f(x,y)=c.
    Draw a handful of levels for a clean topographic look.
    """
    curves = VGroup()
    for c in levels:
        curve = ImplicitFunction(
            lambda x, y, c=c: styblinski_tang_fn(x, y) - c,
            x_range=(x_range[0], x_range[1]),
            y_range=(y_range[0], y_range[1]),
            color=color,
            stroke_opacity=0.7,
            stroke_width=2.5,
        )
        curves.add(curve)
    return curves


def make_contours_xy_and_on_surface(
    axes: ThreeDAxes,
    levels: list[float],
    x_range=X_RANGE,
    y_range=Y_RANGE,
    color=GREY_B,
) -> tuple[VGroup, VGroup]:
    """
    Returns:
      (contours_xy, contours_3d)
      - contours_xy: contours placed in the axes' XY-plane (z=0), correctly scaled to axes
      - contours_3d: same geometry lifted to z=c (i.e., intersection with z=c planes)
    """
    contours_xy = VGroup()
    contours_3d = VGroup()

    for c in levels:
        # Base 2D curve in (x,y): f(x,y) = c
        base = ImplicitFunction(
            lambda x, y, c=c: styblinski_tang_fn(x, y) - c,
            x_range=(x_range[0], x_range[1]),
            y_range=(y_range[0], y_range[1]),
            color=color,
            stroke_opacity=0.85,
            stroke_width=2.5,
        )
        # Map curve points into the axes' XY-plane (z=0), so it aligns with the axes scaling
        base.apply_function(lambda p: axes.c2p(p[0], p[1], 0.0))
        contours_xy.add(base)

        # Lift that curve onto the surface height z=c (because f(x,y)=c on the curve)
        lifted = base.copy()
        lifted.apply_function(lambda p, c=c: axes.c2p(*axes.p2c(p)[:2], c))
        contours_3d.add(lifted)

    return contours_xy, contours_3d
