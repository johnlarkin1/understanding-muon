from __future__ import annotations

import numpy as np
from manim import (
    BLUE_B,
    BLUE_D,
    BLUE_E,
    BOLD,
    DEGREES,
    DOWN,
    GREY_B,
    IN,
    LEFT,
    PURPLE_E,
    TAU,
    TEAL_B,
    TEAL_E,
    UL,
    UP,
    UR,
    YELLOW,
    YELLOW_B,
    Arrow3D,
    Axes,
    Dot,
    FadeIn,
    ImplicitFunction,
    MathTex,
    Scene,
    Sphere,
    Surface,
    Tex,
    Text,
    ThreeDAxes,
    ThreeDScene,
    TracedPath,
    Transform,
    VGroup,
    Write,
    interpolate_color,
    smooth,
)

# =============================
# Problem setup (math utilities)
# =============================

# Domain and ranges
X_RANGE = (-5, 5, 1)
Y_RANGE = (-5, 5, 1)
# Over the [-5,5]^2 box, z ≈ [-78.33, ~250]
Z_RANGE = (-100, 100, 50)

GLOBAL_MIN_COORD = -2.903534  # 1D optimum
GLOBAL_MIN_2D = np.array([GLOBAL_MIN_COORD, GLOBAL_MIN_COORD], dtype=float)

# =============================
# Camera params
# =============================
ZOOM = 0.55


def styblinski_tang_fn(x: float, y: float) -> float:
    """Styblinski–Tang function in 2D."""
    return 0.5 * ((x**4 - 16 * x**2 + 5 * x) + (y**4 - 16 * y**2 + 5 * y))


def styblinski_tang_grad(x: float, y: float) -> np.ndarray:
    """Gradient of 2D Styblinski–Tang."""
    dfx = 2 * x**3 - 16 * x + 2.5
    dfy = 2 * y**3 - 16 * y + 2.5
    return np.array([dfx, dfy], dtype=float)


# Precompute a starting point that:
# - plain SGD/GD -> falls into the positive local minimum near (2.746..., 2.746...)
# - SGD with momentum -> escapes to the global minimum (-2.903..., -2.903...)
START = np.array([3.8, 3.8], dtype=float)


# -----------------
# Optimizer helpers
# -----------------
def sgd_path(
    start: np.ndarray, lr: float = 0.02, steps: int = 40, noise_std: float = 0.0
) -> np.ndarray:
    """Vanilla (stochastic) gradient descent path in R^2."""
    xs = [start.astype(float)]
    p = start.astype(float).copy()
    rng = np.random.default_rng(42)
    for _ in range(steps):
        g = styblinski_tang_grad(p[0], p[1])
        if noise_std > 0:
            g = g + rng.normal(0.0, noise_std, size=2)
        p = p - lr * g
        xs.append(p.copy())
    return np.array(xs)


def momentum_path(
    start: np.ndarray, lr: float = 0.05, beta: float = 0.8, steps: int = 60
) -> np.ndarray:
    """Heavy-ball momentum (Polyak)."""
    xs = [start.astype(float)]
    x = start.astype(float).copy()
    v = np.zeros_like(x)
    for _ in range(steps):
        g = styblinski_tang_grad(x[0], x[1])
        v = beta * v - lr * g
        x = x + v
        xs.append(x.copy())
    return np.array(xs)


# Reference paths used in Scene 2 and Scene 3
SGD_PATH = sgd_path(START, lr=0.02, steps=40, noise_std=0.0)
MOM_PATH = momentum_path(START, lr=0.05, beta=0.8, steps=60)

# ============================
# Reusable drawing/scene utils
# ============================


def make_3d_axes() -> ThreeDAxes:
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


def ball_at(axes: ThreeDAxes, p: np.ndarray, radius: float = 0.15, color=YELLOW) -> Sphere:
    """Create a sphere located on the surface at (x,y)."""
    x, y = float(p[0]), float(p[1])
    z = styblinski_tang_fn(x, y)
    s = Sphere(radius=radius, color=color, resolution=(24, 24))
    s.set_shade_in_3d(True)
    s.move_to(axes.c2p(x, y, z + radius))  # slightly above surface
    return s


def grad_arrow_3d(axes: ThreeDAxes, p: np.ndarray, scale_xy: float = 0.5) -> Arrow3D:
    """
    Arrow showing steepest descent direction (projected onto surface tangent).
    The XY projection follows -grad; Z component decreases accordingly.
    """
    x, y = float(p[0]), float(p[1])
    g = styblinski_tang_grad(x, y)
    norm = np.linalg.norm(g)
    if norm < 1e-9:
        gdir = np.zeros(2)
    else:
        gdir = -g / norm
    # small step in parameter space to visualize tangent direction
    dx, dy = (scale_xy * gdir[0], scale_xy * gdir[1])
    dz = styblinski_tang_fn(x + dx, y + dy) - styblinski_tang_fn(x, y)
    start = axes.c2p(x, y, styblinski_tang_fn(x, y) + 0.2)
    end = axes.c2p(x + dx, y + dy, styblinski_tang_fn(x, y) + dz + 0.2)
    return Arrow3D(start=start, end=end, color=YELLOW_B, stroke_width=6)


def grad_panel(p: np.ndarray) -> VGroup:
    x, y = float(p[0]), float(p[1])
    g = styblinski_tang_grad(x, y)
    txt = VGroup(
        MathTex(r"f(x,y)=\tfrac12\big[(x^4-16x^2+5x)+(y^4-16y^2+5y)\big]").scale(0.5),
        MathTex(r"\nabla f(x,y)=\big(2x^3-16x+2.5,\;2y^3-16y+2.5\big)").scale(0.5),
        MathTex(rf"x={x:.3f},\;y={y:.3f}").scale(0.5),
        MathTex(rf"\nabla f=({g[0]:.3f},\;{g[1]:.3f})").scale(0.5),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
    txt.to_corner(UR).set_background_stroke(opacity=0).set_fill(opacity=1)
    txt.set_color_by_gradient(BLUE_B, TEAL_B)
    return txt


def update_panel(panel: VGroup, p: np.ndarray) -> VGroup:
    """Return a new panel (cheap to recreate) to avoid fragile in-place MathTex edits."""
    new_panel = grad_panel(p)
    new_panel.move_to(panel.get_center())
    return new_panel


def make_contours(levels: list[float], x_range=X_RANGE, y_range=Y_RANGE, color=GREY_B) -> VGroup:
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
            stroke_opacity=0.5,
            stroke_width=2,
        )
        curves.add(curve)
    return curves


def animate_path_on_surface(
    scene: ThreeDScene,
    axes: ThreeDAxes,
    surface: Surface,
    path: np.ndarray,
    title: str,
    show_final_note: str | None = None,
    step_time: float = 0.2,
):
    """Animate a path on the 3D surface with a rolling ball and gradient arrows."""
    scene.add(axes, surface)

    title_m = Tex(title).to_edge(UL).scale(0.7)
    # Pin title to screen space (HUD)
    scene.add_fixed_in_frame_mobjects(title_m)
    scene.play(FadeIn(title_m, shift=DOWN, lag_ratio=0.1))

    ball = ball_at(axes, path[0])
    trail = TracedPath(ball.get_center, stroke_width=4, stroke_color=YELLOW, stroke_opacity=0.8)
    panel = grad_panel(path[0])
    # Pin gradient panel to screen space (HUD)
    scene.add_fixed_in_frame_mobjects(panel)

    arr = grad_arrow_3d(axes, path[0], scale_xy=0.6)
    scene.add(trail, ball, panel, arr)

    for i in range(1, len(path)):
        p = path[i]
        # Update panel and gradient arrow
        new_panel = update_panel(panel, p)
        # Keep new panel fixed to screen space before transform
        scene.add_fixed_in_frame_mobjects(new_panel)
        new_arr = grad_arrow_3d(axes, p, scale_xy=0.6)
        # Move ball
        new_pos = axes.c2p(p[0], p[1], styblinski_tang_fn(p[0], p[1]) + ball.radius)
        scene.play(
            ball.animate.move_to(new_pos),
            Transform(panel, new_panel),
            Transform(arr, new_arr),
            run_time=step_time,
            rate_func=smooth,
        )

    if show_final_note:
        note = Tex(show_final_note).scale(0.8).next_to(ball, UP * 2)
        note.set_color(YELLOW_B)
        # Billboard the note in 3D space (positioned relative to ball)
        scene.add_fixed_orientation_mobjects(note)
        scene.play(Write(note))
        scene.wait(0.8)


def animate_path_on_contours(
    scene: Scene,
    path: np.ndarray,
    title: str,
    levels: list[float] = (-70, -60, -50, -40, -30, -20, -10, 0, 20, 50, 100),
    step_time: float = 0.15,
):
    """Animate same path on a 2D contour (topographic) map."""
    ax = Axes(x_range=X_RANGE, y_range=Y_RANGE, x_length=8, y_length=8, tips=False)
    ax_labels = VGroup(ax.get_x_axis_label(Tex("x")), ax.get_y_axis_label(Tex("y")))
    contours = make_contours(list(levels), color=GREY_B)
    # Place contours in the same coordinates as Axes
    # ImplicitFunction is already in scene coordinates; wrap/group with axes origin.
    plot_group = VGroup(contours).move_to(ax.get_origin())

    title_m = Tex(title).to_edge(UL).scale(0.7)

    scene.play(FadeIn(ax), FadeIn(ax_labels))
    scene.play(FadeIn(plot_group, shift=0.25 * UP, lag_ratio=0.05))
    scene.play(FadeIn(title_m, shift=DOWN, lag_ratio=0.1))

    dot = Dot(ax.c2p(path[0, 0], path[0, 1]), color=YELLOW)
    top_trail = TracedPath(dot.get_center, stroke_color=YELLOW, stroke_width=4)
    scene.add(dot, top_trail)

    for i in range(1, len(path)):
        p = path[i]
        scene.play(
            dot.animate.move_to(ax.c2p(float(p[0]), float(p[1]))),
            run_time=step_time,
            rate_func=smooth,
        )

    scene.wait(0.5)


# ============
# Scene 1: 3D
# ============
class ST_SurfaceIntro(ThreeDScene):
    """
    Scene 1 — Visualizing the Styblinski–Tang surface and rotating 360°.
    """

    def construct(self):
        axes, labels = make_3d_axes()
        surface = make_surface(axes)
        # Start at eye-level to see into the valleys
        self.set_camera_orientation(phi=90 * DEGREES, theta=-45 * DEGREES, zoom=ZOOM)

        title = Tex("Styblinski–Tang Function (2D)").to_edge(UL).scale(0.8)
        subtitle = (
            MathTex(r"z = 0.5\,\big[(x^4 - 16x^2 + 5x) + (y^4 - 16y^2 + 5y)\big]")
            .scale(0.6)
            .next_to(title, DOWN, buff=0.25)
        )

        # Billboard the axis labels to face the camera
        self.add_fixed_orientation_mobjects(labels)
        self.play(FadeIn(axes), FadeIn(labels))
        self.play(FadeIn(surface, shift=0.5 * IN), run_time=1.2)

        # Pin title and subtitle to screen space (HUD)
        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(0.5)

        # First 360° rotation at eye level (phi=90°)
        T = 12
        self.begin_ambient_camera_rotation(rate=TAU / T)
        self.wait(T)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)

        # Raise camera to overhead view to see bowl shape
        self.move_camera(phi=70 * DEGREES, theta=-45 * DEGREES, run_time=2.0)
        self.wait(0.5)

        # Second 360° rotation at overhead angle (phi=70°)
        self.begin_ambient_camera_rotation(rate=TAU / T)
        self.wait(T)
        self.stop_ambient_camera_rotation()

        prompt = Text("How should we walk down this surface?", weight=BOLD).scale(0.6)
        prompt.to_edge(DOWN)
        # Pin prompt to screen space (HUD)
        self.add_fixed_in_frame_mobjects(prompt)
        self.play(Write(prompt))
        self.wait(0.8)
