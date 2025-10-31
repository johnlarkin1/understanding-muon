import numpy as np
from manim import (
    BLACK,
    BLUE_B,
    BLUE_D,
    BLUE_E,
    BOLD,
    DEGREES,
    DL,
    DOWN,
    GRAY_D,
    GREEN,
    GREY_B,
    IN,
    LEFT,
    ORIGIN,
    PURPLE_E,
    RED,
    RIGHT,
    TAU,
    TEAL_B,
    TEAL_E,
    UL,
    UP,
    WHITE,
    YELLOW_E,
    Arrow,
    Axes,
    Create,
    DashedLine,
    Dot,
    Ellipse,
    FadeIn,
    FadeOut,
    ImplicitFunction,
    LaggedStart,
    MathTex,
    Matrix,
    Paragraph,
    ReplacementTransform,
    RoundedRectangle,
    Scene,  # <-- added
    Surface,
    Tex,
    Text,
    ThreeDAxes,
    ThreeDScene,
    VGroup,
    Write,
    interpolate_color,
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
    """Styblinski–Tang function in 2D."""
    return 0.5 * ((x**4 - 16 * x**2 + 5 * x) + (y**4 - 16 * y**2 + 5 * y))


def styblinski_tang_grad(x: float, y: float) -> np.ndarray:
    """Gradient of 2D Styblinski–Tang."""
    dfx = 2 * x**3 - 16 * x + 2.5
    dfy = 2 * y**3 - 16 * y + 2.5
    return np.array([dfx, dfy], dtype=float)


def styblinski_tang_hessian(x: float, y: float) -> np.ndarray:
    """Hessian matrix of 2D Styblinski–Tang.

    Returns 2x2 matrix of second derivatives.
    For Styblinski-Tang, the function is separable so cross-derivatives are zero.
    """
    d2f_dx2 = 6 * x**2 - 16
    d2f_dy2 = 6 * y**2 - 16
    return np.array([[d2f_dx2, 0.0], [0.0, d2f_dy2]], dtype=float)


# Precompute a starting point that:
# - plain SGD/GD -> falls into the positive local minimum near (2.746..., 2.746...)
# - SGD with momentum -> escapes to the global minimum (-2.903..., -2.903...)
START = np.array([3.8, 3.8], dtype=float)
LOCAL_MINIMA = [
    np.array([2.746, 2.746]),
    np.array([-2.903, -2.903]),
    np.array([-2.903, -2.903]),
]

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


def make_muon_algorithm_panel() -> tuple[VGroup, list[MathTex]]:
    """(Hoisted from the class so both scenes can reuse it.)"""
    title = Text("Muon update (per step)", weight=BOLD).scale(0.38)
    lines = [
        r"1.\quad g_t \leftarrow \nabla_\theta f(\theta_{t-1})",
        r"2.\quad B_t \leftarrow \mu\,B_{t-1} + g_t",
        r"3.\quad \widetilde{B}_t \leftarrow \begin{cases} g_t + \mu B_t, & \text{Nesterov} \\ B_t, & \text{otherwise} \end{cases}",
        r"4.\quad O_t \leftarrow \mathrm{NewtonSchulz}^{(a,b,c)}_{k}(\widetilde{B}_t;\,\varepsilon)",
        r"5.\quad \theta_t \leftarrow \theta_{t-1} - \gamma\,\lambda\,\theta_{t-1} \quad\small\text{(decoupled decay)}",
        r"6.\quad \gamma \leftarrow 0.2\,\gamma \sqrt{\max(A,B)} \quad\small\text{(Moonshot adjust)}",
        r"7.\quad \theta_t \leftarrow \theta_{t} - \gamma\,O_t",
    ]
    tex = [MathTex(s).scale(0.42) for s in lines]
    stack = VGroup(*tex).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    panel = VGroup(title, stack).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
    panel.to_edge(RIGHT, buff=0.3)

    bg = RoundedRectangle(
        corner_radius=0.12, width=5.8, height=panel.height + 0.35, stroke_width=1.5
    )
    bg.set_fill(BLACK, opacity=0.78).set_stroke(GRAY_D, width=1.5)
    group = VGroup(bg, panel)
    bg.surround(panel, stretch=True, buff=0.18)
    return group, tex


# =============================
# Scene 1 (3D): everything up to the 2D handoff
# =============================
class MuonOverview3D(ThreeDScene):
    def construct(self):
        # Section 1: Surface introduction
        self.next_section("surface_intro", skip_animations=False)

        axes, labels = make_3d_axes()
        surface = make_surface(axes)
        self.set_camera_orientation(phi=70 * DEGREES, theta=-450 * DEGREES, zoom=ZOOM)

        title = Tex("Styblinski–Tang Function (2D)").to_edge(UL).scale(0.8)
        subtitle = (
            MathTex(r"z = 0.5\,\big[(x^4 - 16x^2 + 5x) + (y^4 - 16y^2 + 5y)\big]")
            .scale(0.6)
            .next_to(title, DOWN, buff=0.25)
        )

        self.add_fixed_orientation_mobjects(labels)
        self.play(FadeIn(axes), FadeIn(labels))
        self.play(FadeIn(surface, shift=0.5 * IN), run_time=1.2)

        # Pin title and subtitle to screen space (HUD)
        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(0.5)

        # Section 2: Show contours on 3D surface
        self.next_section("show_contours", skip_animations=False)

        # Introduce contour visualization
        contour_intro = Text("let's visualize contour lines").scale(0.6)
        contour_intro.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(contour_intro)
        self.play(Write(contour_intro))
        self.wait(0.5)

        # Create contours in 3D space (at their actual z=c heights) with different colors
        contour_levels = [-70, -50, -30, -10, 10, 30, 50, 80]
        contour_colors = [
            RED,
            interpolate_color(RED, PURPLE_E, 0.3),
            PURPLE_E,
            interpolate_color(PURPLE_E, BLUE_E, 0.3),
            BLUE_E,
            interpolate_color(BLUE_E, TEAL_E, 0.3),
            TEAL_E,
            GREEN,
        ]

        contours_3d = VGroup()
        for c, color in zip(contour_levels, contour_colors):
            base = ImplicitFunction(
                lambda x, y, c=c: styblinski_tang_fn(x, y) - c,
                x_range=(X_RANGE[0], X_RANGE[1]),
                y_range=(Y_RANGE[0], Y_RANGE[1]),
                color=color,
                stroke_opacity=1.0,
                stroke_width=3.5,
            )
            # Lift curve onto the surface height z=c
            base.apply_function(lambda p, c=c: axes.c2p(p[0], p[1], c))
            contours_3d.add(base)

        self.play(LaggedStart(*[FadeIn(c) for c in contours_3d], lag_ratio=0.1), run_time=2.0)
        self.wait(0.5)
        self.play(FadeOut(contour_intro, shift=0.3 * DOWN))
        self.wait(0.5)

        # Section 3: Rotate camera 360°
        self.next_section("rotate_view", skip_animations=False)

        T = 6
        self.begin_ambient_camera_rotation(rate=TAU / T)
        self.wait(T)
        self.stop_ambient_camera_rotation()
        self.wait(0.3)

        # Section 4: Flatten to 2D projection
        self.next_section("flatten_to_2d", skip_animations=False)

        prompt = Text("projecting onto 2D").scale(0.6)
        prompt.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(prompt)
        self.play(Write(prompt))
        self.wait(0.8)

        # Flatten the surface while still viewing from 3D angle
        flattened_surface = surface.copy()
        flattened_surface.apply_function(lambda p: np.array([p[0], p[1], 0.02 * p[2]]))
        flattened_surface.set_fill(opacity=0.75)
        flattened_surface.set_stroke(width=0.3, opacity=0.35)

        # project 3d contours onto 2d plane
        self.play(
            ReplacementTransform(surface, flattened_surface),
            FadeOut(prompt, shift=0.3 * DOWN),
            FadeOut(labels),  # Hide z-axis label when going to 2D
            contours_3d.animate.apply_function(lambda p: axes.c2p(*axes.p2c(p)[:2], 0.0)),
            run_time=2.0,
        )
        surface = flattened_surface
        contours = contours_3d

        # Top-down camera
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, gamma=0, run_time=1.5)
        self.wait(0.4)

        # Section 5: Show local and global minima
        self.next_section("show_minima", skip_animations=False)

        local_markers = VGroup()
        local_labels = VGroup()
        for i, coord in enumerate(LOCAL_MINIMA_COORDS):
            dot = Dot(axes.c2p(coord[0], coord[1], 0), radius=0.07, color=RED)
            label = Text(f"Local {i + 1}", weight=BOLD).scale(0.32).set_color(RED)
            label.next_to(dot, UP + RIGHT * 0.5, buff=0.15)
            local_markers.add(dot)
            local_labels.add(label)

        global_marker = Dot(
            axes.c2p(GLOBAL_MIN_2D[0], GLOBAL_MIN_2D[1], 0), radius=0.09, color=GREEN
        )
        global_label = Text("Global min", weight=BOLD).scale(0.38).set_color(GREEN)
        global_label.next_to(global_marker, DOWN + LEFT * 0.3, buff=0.18)

        self.add_fixed_orientation_mobjects(local_labels, global_label)
        self.play(
            LaggedStart(
                *[FadeIn(dot, scale=0.8) for dot in local_markers], lag_ratio=0.2, run_time=1.2
            ),
            FadeIn(global_marker, scale=1.3),
            FadeIn(local_labels, shift=0.2 * UP),
            FadeIn(global_label, shift=0.2 * UP),
        )
        self.wait(0.6)

        # Section 6: Add algorithm panel and reposition
        self.next_section("add_algorithm_panel", skip_animations=False)

        contour_plane = VGroup(
            axes,
            surface,
            contours,
            local_markers,
            global_marker,
            local_labels,
            global_label,
        )

        panel_group, _panel_lines_unused = make_muon_algorithm_panel()
        self.add_fixed_in_frame_mobjects(panel_group)
        self.play(FadeIn(panel_group, shift=0.3 * LEFT), run_time=1.4)
        self.wait(0.5)

        self.play(contour_plane.animate.scale(0.88).shift(LEFT * 6.5), run_time=1.8)
        self.wait(0.4)

        # ---------- HANDOFF: end Scene 1 with blank left side + panel visible ----------
        self.next_section("handoff_to_2d_scene", skip_animations=False)
        self.play(FadeOut(contour_plane), run_time=0.8)
        self.wait(0.2)
        # End of Scene 1


# =============================
# Scene 2 (2D): everything from "STEP ONE - gradient calc" onward
# =============================
class MuonGradient2D(Scene):
    def construct(self):
        # Recreate the algorithm panel in the same place so the first frame
        # matches the last frame of the 3D scene (seamless cut).
        panel_group, panel_lines = make_muon_algorithm_panel()
        self.add(panel_group)

        # Add title and subtitle to match the ending of the 3D scene
        title = Tex("Styblinski–Tang Function (2D)").to_edge(UL).scale(0.8)
        formula = (
            MathTex(r"z = 0.5\,\big[(x^4 - 16x^2 + 5x) + (y^4 - 16y^2 + 5y)\big]")
            .scale(0.6)
            .next_to(title, DOWN, buff=0.25)
        )
        self.add(title, formula)
        self.wait(0.5)

        # Transform title text into formula and move to top
        formula_top = formula.copy().to_edge(UL, buff=0.3)
        self.play(FadeOut(title), ReplacementTransform(formula, formula_top), run_time=0.8)
        self.wait(0.3)

        #########################################################
        # STEP ONE - gradient calc
        #########################################################
        self.next_section("gradient_calculation", skip_animations=False)

        # (No contour_plane to fade here; it was faded at the end of Scene 1.)

        # pick single point for example step
        theta_start = np.array([4.1, 4.5], dtype=float)
        local_axes = Axes(
            x_range=[theta_start[0] - 3.5, theta_start[0] + 1.5, 1.0],
            y_range=[theta_start[1] - 3.5, theta_start[1] + 1.5, 1.0],
            x_length=4.5,
            y_length=4.5,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 1.5},
        )
        local_axes.to_edge(LEFT, buff=0.6).shift(UP * 0.3)

        # matched contour levels/colors (same as 3D)
        contour_levels = [-70, -50, -30, -10, 10, 30, 50, 80]
        contour_colors = [
            RED,
            interpolate_color(RED, PURPLE_E, 0.3),
            PURPLE_E,
            interpolate_color(PURPLE_E, BLUE_E, 0.3),
            BLUE_E,
            interpolate_color(BLUE_E, TEAL_E, 0.3),
            TEAL_E,
            GREEN,
        ]

        local_contours = VGroup()
        for c, color in zip(contour_levels, contour_colors):
            curve = ImplicitFunction(
                lambda x, y, c=c: styblinski_tang_fn(x, y) - c,
                x_range=(theta_start[0] - 3.5, theta_start[0] + 3.5),
                y_range=(theta_start[1] - 3.5, theta_start[1] + 3.5),
                color=color,
                stroke_opacity=0.8,
                stroke_width=2.5,
            )
            curve.apply_function(lambda p: local_axes.c2p(p[0], p[1]))
            local_contours.add(curve)

        # θ₁ point and label
        theta1_dot = Dot(local_axes.c2p(theta_start[0], theta_start[1]), radius=0.07, color=BLUE_E)
        theta1_label = (
            MathTex(r"\theta_1").scale(0.6).next_to(theta1_dot, UP * 2 + RIGHT * 0.35, buff=0.15)
        )
        theta1_sublabel = (
            MathTex(r"(4.1,\ 4.5)")
            .scale(0.45)
            .set_color(BLUE_B)
            .next_to(theta1_label, DOWN, buff=0.08)
        )

        note = (
            Text("Starting at step t=1 (not t=0)\nso we can show momentum effects", font_size=20)
            .set_color(GREY_B)
            .to_edge(DOWN, buff=0.3)
        )
        self.play(FadeIn(local_axes), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(c) for c in local_contours], lag_ratio=0.15),
            run_time=1.5,
        )

        self.add(theta1_dot, theta1_label, theta1_sublabel, note)
        self.play(
            FadeIn(theta1_dot, scale=1.2),
            FadeIn(theta1_label, shift=0.1 * UP),
            FadeIn(theta1_sublabel, shift=0.05 * DOWN),
            FadeIn(note),
            run_time=0.6,
        )
        self.wait(0.8)
        self.play(FadeOut(note), run_time=0.4)
        self.wait(0.3)

        #########################
        # run through a single step of the muon optimizer
        #########################

        # show analytical form of gradient for our loss function, highlight step 1
        step1 = panel_lines[0]  # "g_t <- ∇_θ f(θ_{t-1})"
        step1_box = RoundedRectangle(
            corner_radius=0.08,
            width=step1.width + 0.30,
            height=step1.height + 0.20,
        )
        step1_box.set_fill(opacity=0).set_stroke(TEAL_B, width=2.4)
        step1_box.move_to(step1)
        self.add(step1_box)
        self.play(FadeIn(step1_box, scale=0.98), step1.animate.set_color(TEAL_B))

        # derivatives walkthrough
        gradient_step1 = MathTex(
            r"f(x, y) = 0.5[(x^4 - 16x^2 + 5x) \\",
            r"+ (y^4 - 16y^2 + 5y)]",
            font_size=30,
        ).move_to(ORIGIN)
        self.play(Write(gradient_step1), run_time=0.8)
        self.wait(1.04)

        gradient_step2 = MathTex(
            r"\frac{\partial f}{\partial x} &= 0.5 \cdot \frac{\partial}{\partial x}(x^4 - 16x^2 + 5x) \\",
            r"&= 0.5(4x^3 - 32x + 5) \\",
            r"&= 2x^3 - 16x + 2.5",
            font_size=30,
        ).move_to(ORIGIN)
        self.play(ReplacementTransform(gradient_step1, gradient_step2), run_time=0.9)
        self.wait(1.04)

        gradient_step3 = MathTex(
            r"\frac{\partial f}{\partial y} &= 0.5 \cdot \frac{\partial}{\partial y}(y^4 - 16y^2 + 5y) \\",
            r"&= 0.5(4y^3 - 32y + 5) \\",
            r"&= 2y^3 - 16y + 2.5",
            font_size=30,
        ).move_to(ORIGIN)
        self.play(ReplacementTransform(gradient_step2, gradient_step3), run_time=0.9)
        self.wait(1.04)

        gradient_step4 = MathTex(
            r"\frac{\partial f}{\partial x}(x,y) &= 2x^3 - 16x + 2.5 \\",
            r"\frac{\partial f}{\partial y}(x,y) &= 2y^3 - 16y + 2.5",
            font_size=30,
        ).move_to(ORIGIN)
        self.play(ReplacementTransform(gradient_step3, gradient_step4), run_time=0.8)
        self.wait(1.04)

        gradient_step5 = MathTex(
            r"\frac{\partial f}{\partial x}(4.1, 4.5) &= 2(4.1)^3 - 16(4.1) + 2.5 \\",
            r"\frac{\partial f}{\partial y}(4.1, 4.5) &= 2(4.5)^3 - 16(4.5) + 2.5",
            font_size=30,
        ).move_to(ORIGIN)
        self.play(ReplacementTransform(gradient_step4, gradient_step5), run_time=0.9)
        self.wait(1.04)

        gx = 2 * (4.1**3) - 16 * 4.1 + 2.5
        gy = 2 * (4.5**3) - 16 * 4.5 + 2.5
        gradient_step6 = MathTex(
            rf"g_1 = \nabla f(\theta_1) = \begin{{pmatrix}} {gx:.2f} \\ {gy:.2f} \end{{pmatrix}}",
            font_size=30,
        ).move_to(ORIGIN)
        gradient_step6[0][-14:].set_color(YELLOW_E)
        self.play(ReplacementTransform(gradient_step5, gradient_step6), run_time=0.8)
        self.wait(1.05)

        # gradient arrows
        grad = styblinski_tang_grad(theta_start[0], theta_start[1])
        scale_factor = 0.2

        gradient_arrow = Arrow(
            start=local_axes.c2p(theta_start[0], theta_start[1]),
            end=local_axes.c2p(
                theta_start[0] + grad[0] * scale_factor, theta_start[1] + grad[1] * scale_factor
            ),
            buff=0,
            color=GREEN,
            stroke_width=5,
        )
        gradient_label = MathTex(r"g_1", color=GREEN).scale(0.6)
        gradient_label.next_to(gradient_arrow.get_end(), UP + RIGHT * 0.3, buff=0.1)

        self.add(gradient_arrow, gradient_label)
        self.play(FadeIn(gradient_arrow, scale=0.8), Write(gradient_label), run_time=0.8)
        self.wait(0.8)

        gradient_step7 = MathTex(
            rf"-g_1 = -\nabla f(\theta_1) = \begin{{pmatrix}} {-gx:.2f} \\ {-gy:.2f} \end{{pmatrix}}",
            font_size=30,
        ).move_to(ORIGIN)
        gradient_step7.set_color(BLUE_B)

        neg_gradient_arrow = Arrow(
            start=local_axes.c2p(theta_start[0], theta_start[1]),
            end=local_axes.c2p(
                theta_start[0] - grad[0] * scale_factor, theta_start[1] - grad[1] * scale_factor
            ),
            buff=0,
            color=BLUE_E,
            stroke_width=5,
        )
        neg_gradient_label = MathTex(r"-g_1", color=BLUE_B).scale(0.6)
        neg_gradient_label.next_to(neg_gradient_arrow.get_end(), DOWN + LEFT * 0.3, buff=0.1)

        self.add(neg_gradient_arrow, neg_gradient_label)
        self.play(
            ReplacementTransform(gradient_step6, gradient_step7),
            ReplacementTransform(gradient_arrow, neg_gradient_arrow),
            ReplacementTransform(gradient_label, neg_gradient_label),
            run_time=1.0,
        )
        self.wait(0.8)

        # clean up step 1
        self.play(
            FadeOut(step1_box),
            panel_lines[0].animate.set_color(WHITE),
            FadeOut(gradient_step7, shift=0.2 * DOWN),
            run_time=0.6,
        )
        self.wait(0.3)

        #########################################################
        # STEP TWO - momentum update
        #########################################################
        self.next_section("momentum_update", skip_animations=False)

        step2 = panel_lines[1]
        step2_box = RoundedRectangle(
            corner_radius=0.08,
            width=step2.width + 0.30,
            height=step2.height + 0.20,
        )
        step2_box.set_fill(opacity=0).set_stroke(TEAL_B, width=2.4)
        step2_box.move_to(step2)

        self.add(step2_box)
        self.play(FadeIn(step2_box, scale=0.98), step2.animate.set_color(TEAL_B), run_time=0.6)
        self.wait(0.3)

        B0 = np.array([gx * 0.4, gy * 1.1], dtype=float)
        mu = 0.9
        momentum_note = (
            Text(
                "Note: Momentum buffer accumulates gradients (positive)\nThe minus sign is applied in the parameter update (step 7)",
                font_size=18,
            )
            .set_color(GREY_B)
            .to_edge(DOWN, buff=0.3)
        )

        momentum_eq1 = MathTex(r"B_1 = \mu \cdot B_0 + g_1", font_size=30).move_to(
            ORIGIN + RIGHT * 0.5
        )

        self.add(momentum_eq1, momentum_note)
        self.play(Write(momentum_eq1), FadeIn(momentum_note), run_time=0.8)
        self.wait(1.0)
        self.play(FadeOut(momentum_note), run_time=0.4)
        self.wait(0.3)

        b0_arrow = Arrow(
            start=local_axes.c2p(theta_start[0], theta_start[1]),
            end=local_axes.c2p(
                theta_start[0] + B0[0] * scale_factor, theta_start[1] + B0[1] * scale_factor
            ),
            buff=0,
            color=GREY_B,
            stroke_width=4,
        )
        b0_label = MathTex(r"B_0", color=GREY_B).scale(0.55)
        b0_label.next_to(b0_arrow.get_end(), UP + LEFT * 0.2, buff=0.1)

        self.add(b0_arrow, b0_label)
        self.play(FadeIn(b0_arrow, scale=0.8), Write(b0_label), run_time=0.7)
        self.wait(0.5)

        momentum_eq2 = MathTex(r"\mu \cdot B_0 = 0.9 \cdot B_0", font_size=30).move_to(
            ORIGIN + RIGHT * 0.5
        )

        mu_b0_arrow = Arrow(
            start=local_axes.c2p(theta_start[0], theta_start[1]),
            end=local_axes.c2p(
                theta_start[0] + mu * B0[0] * scale_factor,
                theta_start[1] + mu * B0[1] * scale_factor,
            ),
            buff=0,
            color=PURPLE_E,
            stroke_width=4.5,
        )
        mu_b0_label = MathTex(r"\mu B_0", font_size=40, color=PURPLE_E)
        mu_b0_label.next_to(mu_b0_arrow.get_end(), UP + LEFT * 0.2, buff=0.1)

        self.play(
            ReplacementTransform(momentum_eq1, momentum_eq2),
            ReplacementTransform(b0_arrow.copy(), mu_b0_arrow),
            Write(mu_b0_label),
            run_time=0.9,
        )
        self.wait(0.5)

        g1_arrow = Arrow(
            start=local_axes.c2p(theta_start[0], theta_start[1]),
            end=local_axes.c2p(
                theta_start[0] + grad[0] * scale_factor, theta_start[1] + grad[1] * scale_factor
            ),
            buff=0,
            color=GREEN,
            stroke_width=4.5,
        )
        g1_label = MathTex(r"g_1", font_size=40, color=GREEN)
        g1_label.next_to(g1_arrow.get_end(), UP + RIGHT * 0.3, buff=0.1)

        self.play(FadeIn(g1_arrow, scale=0.8), Write(g1_label), run_time=0.7)
        self.wait(0.5)

        B1 = mu * B0 + grad
        momentum_eq3 = MathTex(
            rf"B_1 = \mu B_0 + g_1 = \begin{{pmatrix}} {B1[0]:.2f} \\ {B1[1]:.2f} \end{{pmatrix}}",
            font_size=30,
        ).move_to(ORIGIN + RIGHT * 0.5)
        momentum_eq3[0][-14:].set_color(YELLOW_E)

        b1_arrow = Arrow(
            start=local_axes.c2p(theta_start[0], theta_start[1]),
            end=local_axes.c2p(
                theta_start[0] + B1[0] * scale_factor, theta_start[1] + B1[1] * scale_factor
            ),
            buff=0,
            color=TEAL_E,
            stroke_width=6,
        )
        b1_label = MathTex(r"B_1", font_size=40, color=TEAL_E).set(weight=BOLD)
        b1_label.next_to(b1_arrow.get_end(), RIGHT * 0.5, buff=0.15)

        self.play(
            ReplacementTransform(momentum_eq2, momentum_eq3),
            FadeIn(b1_arrow, scale=0.9),
            Write(b1_label),
            run_time=1.0,
        )
        self.wait(1.2)

        self.play(
            FadeOut(b0_arrow),
            FadeOut(b0_label),
            FadeOut(mu_b0_arrow),
            FadeOut(mu_b0_label),
            FadeOut(g1_arrow),
            FadeOut(g1_label),
            FadeOut(neg_gradient_arrow),
            FadeOut(neg_gradient_label),
            FadeOut(momentum_eq3),
            run_time=0.8,
        )
        self.wait(0.5)

        # clean up step 2
        self.play(
            FadeOut(step2_box),
            panel_lines[1].animate.set_color(WHITE),
            run_time=0.6,
        )
        self.wait(0.3)

        #########################################################
        # STEP THREE - Nesterov momentum (modified buffer)
        #########################################################
        self.next_section("nesterov_momentum", skip_animations=False)

        step3 = panel_lines[2]
        step3_box = RoundedRectangle(
            corner_radius=0.08,
            width=step3.width + 0.30,
            height=step3.height + 0.20,
        )
        step3_box.set_fill(opacity=0).set_stroke(TEAL_B, width=2.4)
        step3_box.move_to(step3)

        self.add(step3_box)
        self.play(FadeIn(step3_box, scale=0.98), step3.animate.set_color(TEAL_B), run_time=0.6)
        self.wait(0.3)

        # Case 1: Non-Nesterov (standard momentum)
        case1_title = Text("Case 1: Standard Momentum (no Nesterov)", font_size=26).set_color(
            GREY_B
        )
        case1_title.to_edge(DOWN, buff=0.5)

        nesterov_eq1 = MathTex(r"\widetilde{B}_1 = B_1", font_size=32).move_to(ORIGIN + RIGHT * 0.5)

        self.add(case1_title)
        self.play(FadeIn(case1_title), Write(nesterov_eq1), run_time=0.8)
        self.wait(0.5)

        # Relabel B_1 arrow as B_tilde_1
        b1_tilde_label_standard = MathTex(r"\widetilde{B}_1", font_size=40, color=TEAL_E).set(
            weight=BOLD
        )
        b1_tilde_label_standard.next_to(b1_arrow.get_end(), RIGHT * 0.5, buff=0.15)

        self.play(
            ReplacementTransform(b1_label, b1_tilde_label_standard),
            run_time=0.7,
        )
        self.wait(0.8)

        # Clean up Case 1
        self.play(
            FadeOut(case1_title),
            FadeOut(nesterov_eq1),
            run_time=0.6,
        )
        self.wait(0.3)

        # Case 2: Nesterov momentum (with detailed breakdown)
        case2_title = Text("Case 2: Nesterov Momentum (lookahead)", font_size=26).set_color(
            YELLOW_E
        )
        case2_title.to_edge(DOWN, buff=0.5)

        nesterov_eq2 = MathTex(r"\widetilde{B}_1 = g_1 + \mu B_1", font_size=32).move_to(
            ORIGIN + RIGHT * 0.5
        )

        self.add(case2_title)
        self.play(FadeIn(case2_title), Write(nesterov_eq2), run_time=0.8)
        self.wait(0.8)

        # Step 1: Recall g_1 value
        nesterov_step1 = MathTex(
            rf"g_1 = \begin{{pmatrix}} {grad[0]:.2f} \\ {grad[1]:.2f} \end{{pmatrix}}",
            font_size=32,
        ).move_to(ORIGIN + RIGHT * 0.5)
        nesterov_step1[0][-14:].set_color(GREEN)

        self.play(ReplacementTransform(nesterov_eq2, nesterov_step1), run_time=0.8)
        self.wait(0.6)

        # Step 2: Show mu (beta) hyperparameter
        mu_note = Text(
            r"μ (mu) = 0.9  [momentum coeff, aka β]",
            font_size=22,
        ).next_to(nesterov_step1, DOWN, buff=0.3)

        self.play(FadeIn(mu_note, shift=0.1 * DOWN), run_time=0.7)
        self.wait(0.8)

        # Step 3: Recall B_1 value
        nesterov_step2 = MathTex(
            rf"B_1 = \begin{{pmatrix}} {B1[0]:.2f} \\ {B1[1]:.2f} \end{{pmatrix}}",
            font_size=32,
        ).move_to(ORIGIN + RIGHT * 0.5)
        nesterov_step2[0][-14:].set_color(TEAL_E)

        self.play(
            FadeOut(nesterov_step1, shift=0.2 * UP),
            FadeIn(nesterov_step2, shift=0.2 * UP),
            run_time=0.8,
        )
        self.wait(0.6)

        # Step 4: Calculate mu * B_1
        mu_B1_nesterov = mu * B1
        nesterov_step3 = MathTex(
            rf"\mu B_1 &= 0.9 \times \begin{{pmatrix}} {B1[0]:.2f} \\ {B1[1]:.2f} \end{{pmatrix}} \\",
            rf"&= \begin{{pmatrix}} {mu_B1_nesterov[0]:.2f} \\ {mu_B1_nesterov[1]:.2f} \end{{pmatrix}}",
            font_size=28,
        ).move_to(ORIGIN + RIGHT * 0.5)
        nesterov_step3[0][-14:].set_color(PURPLE_E)

        self.play(
            FadeOut(nesterov_step2, shift=0.2 * UP),
            FadeIn(nesterov_step3, shift=0.2 * UP),
            run_time=0.9,
        )
        self.wait(0.8)

        # Step 5: Show the substitution into the formula
        nesterov_step4 = MathTex(
            r"\widetilde{B}_1 = g_1 + \mu B_1",
            font_size=32,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(
            FadeOut(nesterov_step3, shift=0.2 * UP),
            FadeOut(mu_note),
            FadeIn(nesterov_step4, shift=0.2 * UP),
            run_time=0.8,
        )
        self.wait(0.5)

        # Step 6: Substitute values
        nesterov_step5 = MathTex(
            rf"\widetilde{{B}}_1 = \begin{{pmatrix}} {grad[0]:.2f} \\ {grad[1]:.2f} \end{{pmatrix}} + \begin{{pmatrix}} {mu_B1_nesterov[0]:.2f} \\ {mu_B1_nesterov[1]:.2f} \end{{pmatrix}}",
            font_size=26,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(ReplacementTransform(nesterov_step4, nesterov_step5), run_time=0.9)
        self.wait(0.6)

        # Step 7: Show visual vector addition with arrows
        # First, show mu * B_1
        mu_b1_arrow_nest = Arrow(
            start=local_axes.c2p(theta_start[0], theta_start[1]),
            end=local_axes.c2p(
                theta_start[0] + mu_B1_nesterov[0] * scale_factor,
                theta_start[1] + mu_B1_nesterov[1] * scale_factor,
            ),
            buff=0,
            color=PURPLE_E,
            stroke_width=4.5,
        )
        mu_b1_label_nest = MathTex(r"\mu B_1", font_size=40, color=PURPLE_E)
        mu_b1_label_nest.next_to(mu_b1_arrow_nest.get_end(), UP + LEFT * 0.2, buff=0.1)

        self.play(FadeIn(mu_b1_arrow_nest, scale=0.8), Write(mu_b1_label_nest), run_time=0.7)
        self.wait(0.5)

        # Show g_1
        g1_arrow_nest = Arrow(
            start=local_axes.c2p(theta_start[0], theta_start[1]),
            end=local_axes.c2p(
                theta_start[0] + grad[0] * scale_factor, theta_start[1] + grad[1] * scale_factor
            ),
            buff=0,
            color=GREEN,
            stroke_width=4.5,
        )
        g1_label_nest = MathTex(r"g_1", font_size=40, color=GREEN)
        g1_label_nest.next_to(g1_arrow_nest.get_end(), UP + RIGHT * 0.3, buff=0.1)

        self.play(FadeIn(g1_arrow_nest, scale=0.8), Write(g1_label_nest), run_time=0.7)
        self.wait(0.5)

        # Step 8: Perform final addition and show result
        B1_tilde_nesterov = grad + mu_B1_nesterov
        nesterov_step6 = MathTex(
            rf"\widetilde{{B}}_1 = \begin{{pmatrix}} {B1_tilde_nesterov[0]:.2f} \\ {B1_tilde_nesterov[1]:.2f} \end{{pmatrix}}",
            font_size=32,
        ).move_to(ORIGIN + RIGHT * 0.5)
        nesterov_step6[0][-14:].set_color(YELLOW_E)

        b1_tilde_arrow_nest = Arrow(
            start=local_axes.c2p(theta_start[0], theta_start[1]),
            end=local_axes.c2p(
                theta_start[0] + B1_tilde_nesterov[0] * scale_factor,
                theta_start[1] + B1_tilde_nesterov[1] * scale_factor,
            ),
            buff=0,
            color=YELLOW_E,
            stroke_width=6,
        )
        b1_tilde_label_nest = MathTex(r"\widetilde{B}_1", font_size=40, color=YELLOW_E).set(
            weight=BOLD
        )
        b1_tilde_label_nest.next_to(b1_tilde_arrow_nest.get_end(), UP + RIGHT * 0.5, buff=0.15)

        self.play(
            ReplacementTransform(nesterov_step5, nesterov_step6),
            FadeIn(b1_tilde_arrow_nest, scale=0.9),
            Write(b1_tilde_label_nest),
            run_time=1.0,
        )
        self.wait(1.2)

        # Clean up intermediate arrows and equations
        self.play(
            FadeOut(mu_b1_arrow_nest),
            FadeOut(mu_b1_label_nest),
            FadeOut(g1_arrow_nest),
            FadeOut(g1_label_nest),
            FadeOut(nesterov_step6),
            FadeOut(case2_title),
            run_time=0.8,
        )
        self.wait(0.5)

        # Side-by-side comparison of both approaches
        comparison_title = Text("Comparison: Standard vs Nesterov", font_size=28, weight=BOLD)
        comparison_title.to_edge(DOWN, buff=0.5)

        # Show both arrows together
        # Make the standard momentum arrow semi-transparent
        self.play(
            FadeIn(comparison_title),
            b1_arrow.animate.set_opacity(0.6),
            b1_tilde_label_standard.animate.set_opacity(0.6),
            run_time=0.7,
        )
        self.wait(0.5)

        # Compute magnitude difference
        mag_standard = np.linalg.norm(B1)
        mag_nesterov = np.linalg.norm(B1_tilde_nesterov)
        percent_increase = ((mag_nesterov - mag_standard) / mag_standard) * 100

        magnitude_comparison = Text(
            f"Nesterov magnitude: {mag_nesterov:.2f} ({percent_increase:.1f}% larger)",
            font_size=22,
        ).set_color(YELLOW_E)
        magnitude_comparison.next_to(comparison_title, UP, buff=0.15)

        self.play(FadeIn(magnitude_comparison, shift=0.1 * UP), run_time=0.8)
        self.wait(1.5)

        # Add annotation about lookahead effect
        lookahead_note = (
            Text("Nesterov 'looks ahead' by amplifying the gradient", font_size=20)
            .set_color(GREY_B)
            .next_to(magnitude_comparison, UP, buff=0.12)
        )
        self.play(FadeIn(lookahead_note, shift=0.1 * UP), run_time=0.6)
        self.wait(1.2)

        # Clean up Step 3 comparison - keep only Nesterov arrow for Step 4
        self.play(
            FadeOut(comparison_title),
            FadeOut(magnitude_comparison),
            FadeOut(lookahead_note),
            FadeOut(b1_arrow),  # Remove standard momentum arrow
            FadeOut(b1_tilde_label_standard),  # Remove standard label
            run_time=0.8,
        )
        self.wait(0.3)

        # Clean up step 3 highlight
        self.play(
            FadeOut(step3_box),
            panel_lines[2].animate.set_color(WHITE),
            run_time=0.6,
        )
        self.wait(0.5)

        # Note: b1_tilde_arrow_nest and b1_tilde_label_nest remain for Step 4 (Newton-Schulz)

        #########################################################
        # STEP FOUR - Newton-Schulz Preconditioning
        #########################################################
        self.next_section("newton_schulz_preconditioning", skip_animations=False)

        step4 = panel_lines[3]
        step4_box = RoundedRectangle(
            corner_radius=0.08,
            width=step4.width + 0.30,
            height=step4.height + 0.20,
        )
        step4_box.set_fill(opacity=0).set_stroke(TEAL_B, width=2.4)
        step4_box.move_to(step4)

        self.add(step4_box)
        self.play(FadeIn(step4_box, scale=0.98), step4.animate.set_color(TEAL_B), run_time=0.6)
        self.wait(0.3)

        # Introduction to curvature-aware preconditioning
        intro_title = Text(
            "Preconditioning: Adjusting for Local Curvature", font_size=26
        ).set_color(GREEN)
        intro_title.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(intro_title), run_time=0.8)
        self.wait(0.8)

        # Compute local Hessian at theta_start to visualize curvature
        H = styblinski_tang_hessian(theta_start[0], theta_start[1])

        # Show the Hessian matrix
        # Create H = label
        h_label = MathTex("H =", font_size=32)

        # Create the matrix with numerical values
        hessian_matrix = Matrix(
            [[f"{H[0, 0]:.1f}", f"{H[0, 1]:.1f}"], [f"{H[1, 0]:.1f}", f"{H[1, 1]:.1f}"]],
            left_bracket="[",
            right_bracket="]",
            element_to_mobject=MathTex,
            h_buff=1.5,
        ).scale(0.6)

        # Group them together
        hessian_display = VGroup(h_label, hessian_matrix).arrange(RIGHT, buff=0.3)
        hessian_display.move_to(ORIGIN + RIGHT * 0.5)

        self.play(
            FadeOut(intro_title),
            Write(hessian_display),
            run_time=1.0,
        )
        self.wait(1.2)

        # Highlight the diagonal structure
        diagonal_note = Text(
            "Diagonal matrix\n→ perpendicular principal axes",
            font_size=20,
        ).set_color(GREY_B)
        diagonal_note.next_to(hessian_display, DOWN, buff=0.25)

        self.play(FadeIn(diagonal_note, shift=0.1 * DOWN), run_time=0.7)
        self.wait(1.0)

        self.play(
            FadeOut(hessian_display, shift=0.2 * UP),
            FadeOut(diagonal_note),
            run_time=0.6,
        )
        self.wait(0.3)

        # Eigendecomposition to get principal axes
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # The Hessian tells us about curvature - larger eigenvalue = steeper curvature
        # For visualization, we'll show the curvature ellipse with fixed base size
        # and aspect ratio determined by eigenvalue ratio

        # Handle sign: eigenvalues can be negative (concave) or positive (convex)
        # For visualization, we care about magnitude
        abs_evals = np.abs(eigenvalues)

        # Use eigenvalue ratio to determine aspect ratio of ellipse
        # The direction with smaller eigenvalue (less curvature) gets larger radius
        min_eval = np.min(abs_evals)
        max_eval = np.max(abs_evals)

        # Base size for visibility (in axes units, will be scaled by scale_factor)
        base_size = 2.5

        # Aspect ratio from eigenvalue ratio (exaggerated for clarity)
        # Use square root to make differences more visible
        raw_ratio = np.sqrt(max_eval / (min_eval + 1e-6))
        aspect_ratio = np.clip(raw_ratio, 1.0, 2.5)

        # Assign radii: larger radius for smaller eigenvalue direction
        if abs_evals[0] < abs_evals[1]:
            radii = np.array([base_size * aspect_ratio, base_size])
        else:
            radii = np.array([base_size, base_size * aspect_ratio])

        # Create ellipse in local coordinates
        curvature_ellipse = Ellipse(
            width=2 * radii[0] * scale_factor,
            height=2 * radii[1] * scale_factor,
            color=PURPLE_E,
            stroke_opacity=0.9,
            stroke_width=4.0,
            fill_opacity=0.25,
            fill_color=PURPLE_E,
        )

        # Position at theta_start
        curvature_ellipse.move_to(local_axes.c2p(theta_start[0], theta_start[1]))

        # Rotate to align with principal axes
        # eigenvectors[:,i] is the i-th eigenvector
        # Angle of first eigenvector (corresponding to eigenvalues[0])
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        curvature_ellipse.rotate(angle, about_point=curvature_ellipse.get_center())

        self.play(
            FadeOut(intro_title),
            Create(curvature_ellipse),
            run_time=1.2,
        )
        self.wait(0.8)

        # Add principal axes as dashed lines
        axis_length = 3.5

        # Easy axis (smaller eigenvalue magnitude = less curvature = easier to move)
        easy_idx = 0 if abs_evals[0] < abs_evals[1] else 1
        hard_idx = 1 - easy_idx

        easy_direction = eigenvectors[:, easy_idx]
        hard_direction = eigenvectors[:, hard_idx]

        easy_axis = DashedLine(
            start=local_axes.c2p(
                theta_start[0] - axis_length * easy_direction[0] * scale_factor,
                theta_start[1] - axis_length * easy_direction[1] * scale_factor,
            ),
            end=local_axes.c2p(
                theta_start[0] + axis_length * easy_direction[0] * scale_factor,
                theta_start[1] + axis_length * easy_direction[1] * scale_factor,
            ),
            color=TEAL_E,
            stroke_width=3.5,
            stroke_opacity=0.9,
            dash_length=0.15,
        )

        hard_axis = DashedLine(
            start=local_axes.c2p(
                theta_start[0] - axis_length * hard_direction[0] * scale_factor,
                theta_start[1] - axis_length * hard_direction[1] * scale_factor,
            ),
            end=local_axes.c2p(
                theta_start[0] + axis_length * hard_direction[0] * scale_factor,
                theta_start[1] + axis_length * hard_direction[1] * scale_factor,
            ),
            color=RED,
            stroke_width=3.5,
            stroke_opacity=0.9,
            dash_length=0.15,
        )

        easy_label = Text("easy axis\n(low curv.)", font_size=16).set_color(TEAL_E)
        easy_label.next_to(easy_axis.get_end(), RIGHT * 0.5, buff=0.1)

        hard_label = Text("hard axis\n(high curv.)", font_size=16).set_color(RED)
        hard_label.next_to(hard_axis.get_end(), UP * 0.5, buff=0.1)

        self.play(
            Create(easy_axis),
            Create(hard_axis),
            Write(easy_label),
            Write(hard_label),
            run_time=1.0,
        )
        self.wait(1.2)
        curvature_note = Text(
            "Low curvature (x-axis)\n-> we can adjust more\nin this direction",
            font_size=18,
            line_spacing=0.8,
        ).set_color(TEAL_E)
        curvature_note.move_to(local_axes.c2p(theta_start[0] + 3.5, theta_start[1] + 1.0))

        self.play(FadeIn(curvature_note, shift=0.1 * RIGHT), run_time=0.8)
        self.wait(1.5)

        # Show the Newton-Schulz formula
        ns_formula = MathTex(
            r"U_1 = \text{NewtonSchulz}(\widetilde{B}_1)",
            font_size=32,
        ).move_to(ORIGIN)

        self.play(Write(ns_formula), run_time=0.8)
        self.wait(0.6)

        # Show the polynomial iteration formula
        ns_detail = MathTex(
            r"\rho(M) = aM + b(MM^T)M + c(MM^T)^2M",
            font_size=26,
        ).next_to(ns_formula, DOWN, buff=0.25)

        params_note = (
            Text(
                "a=3.4445, b=-4.7775, c=2.0315",
                font_size=18,
            )
            .set_color(GREY_B)
            .next_to(ns_detail, DOWN, buff=0.15)
        )

        self.play(
            Write(ns_detail),
            FadeIn(params_note, shift=0.1 * DOWN),
            run_time=0.9,
        )
        self.wait(0.8)

        # For simplification, we'll simulate the preconditioning effect
        # In reality, Newton-Schulz approximates (G^T G)^{-1/2} which rotates and rescales
        # For this visualization, we'll show the direction rotating toward the easy axis
        # and the magnitude being adjusted

        # The preconditioned direction should align more with the easy axis
        # and have adjusted magnitude for safer steps

        # Simulate preconditioned direction: blend toward easy axis and normalize
        precond_direction = (
            0.6 * B1_tilde_nesterov / np.linalg.norm(B1_tilde_nesterov) + 0.4 * easy_direction
        )
        precond_direction = precond_direction / np.linalg.norm(precond_direction)

        # Scale to reasonable magnitude (similar to original for vis clarity)
        O1 = precond_direction * np.linalg.norm(B1_tilde_nesterov) * 0.9

        # Clean up formulas
        self.play(
            FadeOut(ns_formula, shift=0.2 * UP),
            FadeOut(ns_detail),
            FadeOut(params_note),
            run_time=0.6,
        )
        self.wait(0.3)

        # Smooth morphing animation from B̃₁ (yellow) to U₁ (green)
        transform_note = Text(
            "Transforming to curvature-aware direction...",
            font_size=22,
        ).set_color(GREEN)
        transform_note.to_edge(DOWN, buff=0.3)

        self.play(FadeIn(transform_note), run_time=0.6)

        # Create the final preconditioned arrow
        o1_arrow = Arrow(
            start=local_axes.c2p(theta_start[0], theta_start[1]),
            end=local_axes.c2p(
                theta_start[0] + O1[0] * scale_factor,
                theta_start[1] + O1[1] * scale_factor,
            ),
            buff=0,
            color=GREEN,
            stroke_width=6,
        )
        o1_label = MathTex(r"O_1", font_size=40, color=GREEN).set(weight=BOLD)
        o1_label.next_to(o1_arrow.get_end(), RIGHT * 0.5 + UP * 0.3, buff=0.15)

        # Morph the yellow Nesterov arrow into green preconditioned arrow
        self.play(
            ReplacementTransform(b1_tilde_arrow_nest, o1_arrow),
            ReplacementTransform(b1_tilde_label_nest, o1_label),
            run_time=1.5,
        )
        self.wait(1.0)

        self.play(FadeOut(transform_note), run_time=0.5)
        self.wait(0.3)

        # Add explanatory note about preconditioning effect
        precond_note = Text(
            "Preconditioning adjusts for local curvature\n→ safer, more direct progress",
            font_size=20,
        ).set_color(GREY_B)
        precond_note.to_edge(DOWN, buff=0.3)

        self.play(FadeIn(precond_note, shift=0.1 * UP), run_time=0.7)
        self.wait(1.5)

        # Clean up curvature visualization
        self.play(
            FadeOut(curvature_ellipse),
            FadeOut(easy_axis),
            FadeOut(hard_axis),
            FadeOut(easy_label),
            FadeOut(hard_label),
            FadeOut(curvature_note),
            FadeOut(precond_note),
            run_time=0.8,
        )
        self.wait(0.5)

        # Clean up step 4 highlight
        self.play(
            FadeOut(step4_box),
            panel_lines[3].animate.set_color(WHITE),
            run_time=0.6,
        )
        self.wait(0.5)

        # Learning rate used in steps 5, 6, and 7
        # PyTorch default: 1e-3 for torch.optim.Muon
        learning_rate = 1e-3

        #########################################################
        # STEP FIVE - Decoupled Weight Decay
        #########################################################
        self.next_section("weight_decay", skip_animations=False)

        step5 = panel_lines[4]
        step5_box = RoundedRectangle(
            corner_radius=0.08,
            width=step5.width + 0.30,
            height=step5.height + 0.20,
        )
        step5_box.set_fill(opacity=0).set_stroke(TEAL_B, width=2.4)
        step5_box.move_to(step5)

        self.add(step5_box)
        self.play(FadeIn(step5_box, scale=0.98), step5.animate.set_color(TEAL_B), run_time=0.6)
        self.wait(0.3)

        decay_title = Text("Step 5: Decoupled Weight Decay", font_size=26).set_color(PURPLE_E)
        decay_title.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(decay_title), run_time=0.6)
        self.wait(0.5)

        # Show the weight decay formula
        decay_eq1 = MathTex(
            r"\theta \leftarrow \theta - \gamma \lambda \theta",
            font_size=32,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(Write(decay_eq1), run_time=0.8)
        self.wait(0.6)

        # PyTorch default: 0.1 for torch.optim.Muon
        lambda_val = 0.1
        lambda_note = (
            MathTex(
                r"\lambda = " + f"{lambda_val}" + r"\quad\small\text{(weight decay)}",
                font_size=22,
            )
            .set_color(GREY_B)
            .next_to(decay_eq1, DOWN, buff=0.3)
        )

        self.play(FadeIn(lambda_note, shift=0.1 * DOWN), run_time=0.7)
        self.wait(0.8)

        # Show current θ₁ value (before decay)
        decay_eq2 = MathTex(
            rf"\theta_1 = \begin{{pmatrix}} {theta_start[0]:.2f} \\ {theta_start[1]:.2f} \end{{pmatrix}}",
            font_size=32,
        ).move_to(ORIGIN + RIGHT * 0.5)
        decay_eq2[0][-14:].set_color(BLUE_E)

        self.play(
            FadeOut(decay_eq1, shift=0.2 * UP),
            FadeOut(lambda_note),
            FadeIn(decay_eq2, shift=0.2 * UP),
            run_time=0.8,
        )
        self.wait(0.6)

        # Calculate decay term: γλθ
        decay_term = learning_rate * lambda_val * theta_start
        decay_eq3 = MathTex(
            rf"\gamma \lambda \theta_1 = {learning_rate} \times {lambda_val} \times \begin{{pmatrix}} {theta_start[0]:.2f} \\ {theta_start[1]:.2f} \end{{pmatrix}}",
            font_size=26,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(ReplacementTransform(decay_eq2, decay_eq3), run_time=0.9)
        self.wait(0.6)

        # Show the decay term result
        decay_eq4 = MathTex(
            rf"\gamma \lambda \theta_1 = \begin{{pmatrix}} {decay_term[0]:.6f} \\ {decay_term[1]:.6f} \end{{pmatrix}}",
            font_size=28,
        ).move_to(ORIGIN + RIGHT * 0.5)
        self.play(ReplacementTransform(decay_eq3, decay_eq4), run_time=0.8)
        self.wait(0.8)

        theta_after_decay = theta_start - decay_term
        decay_eq5 = MathTex(
            r"\theta_1 \leftarrow \theta_1 - \gamma \lambda \theta_1 \\",
            rf"= \begin{{pmatrix}} {theta_after_decay[0]:.5f} \\ {theta_after_decay[1]:.5f} \end{{pmatrix}}",
            font_size=26,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(ReplacementTransform(decay_eq4, decay_eq5), run_time=0.9)
        self.wait(0.8)

        # Add explanatory note
        decay_note = (
            MathTex(
                r"\text{Regularization: shrinks parameters } \to 0",
                font_size=20,
            )
            .set_color(GREY_B)
            .next_to(decay_eq5, DOWN, buff=0.3)
        )

        self.play(FadeIn(decay_note, shift=0.1 * DOWN), run_time=0.7)
        self.wait(1.2)

        # Clean up step 5
        self.play(
            FadeOut(decay_eq5, shift=0.2 * DOWN),
            FadeOut(decay_note),
            FadeOut(decay_title),
            run_time=0.6,
        )
        self.wait(0.3)

        self.play(
            FadeOut(step5_box),
            panel_lines[4].animate.set_color(WHITE),
            run_time=0.6,
        )
        self.wait(0.5)

        #########################################################
        # STEP SIX - Moonshot Learning Rate Adjustment
        #########################################################
        self.next_section("moonshot_lr_adjustment", skip_animations=False)

        step6 = panel_lines[5]
        step6_box = RoundedRectangle(
            corner_radius=0.08,
            width=step6.width + 0.30,
            height=step6.height + 0.20,
        )
        step6_box.set_fill(opacity=0).set_stroke(TEAL_B, width=2.4)
        step6_box.move_to(step6)

        self.add(step6_box)
        self.play(FadeIn(step6_box, scale=0.98), step6.animate.set_color(TEAL_B), run_time=0.6)
        self.wait(0.3)

        lr_title = Text("Step 6: Moonshot Learning Rate Adjustment", font_size=26).set_color(
            YELLOW_E
        )
        lr_title.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(lr_title), run_time=0.6)
        self.wait(0.5)

        # Show the learning rate adjustment formula
        lr_eq1 = MathTex(
            r"\gamma \leftarrow 0.2 \gamma \sqrt{\max(A, B)}",
            font_size=32,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(Write(lr_eq1), run_time=0.8)
        self.wait(0.6)

        ab_note = (
            Paragraph(
                "A, B = dimensions of weight matrix",
                "(shape AxB)",
                alignment="center",
                font_size=20,
            )
            .set_color(GREY_B)
            .next_to(lr_eq1, DOWN, buff=0.3)
        )

        self.play(FadeIn(ab_note, shift=0.1 * DOWN), run_time=0.7)
        self.wait(1.0)

        A_val = 512
        B_val = 768
        lr_eq2 = MathTex(
            rf"A = {A_val}, \quad B = {B_val} \\ \text{{(example values)}}",
            font_size=28,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(
            FadeOut(lr_eq1, shift=0.2 * UP),
            FadeOut(ab_note),
            FadeIn(lr_eq2, shift=0.2 * UP),
            run_time=0.8,
        )
        self.wait(0.6)

        # Calculate max(A, B)
        max_ab = max(A_val, B_val)
        lr_eq3 = MathTex(
            r"\begin{aligned}"
            rf"&\max(A, B) &&= \max({A_val}, {B_val}) \\"  # first line
            rf"&                &&= {max_ab}"
            r"\end{aligned}",
            font_size=30,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(ReplacementTransform(lr_eq2, lr_eq3), run_time=0.8)
        self.wait(0.6)

        # Calculate square root
        sqrt_max_ab = np.sqrt(max_ab)
        lr_eq4 = MathTex(
            rf"\sqrt{{\max(A, B)}} &= \sqrt{{{max_ab}}} \\"
            rf"&\approx {sqrt_max_ab:.3f}",
            font_size=30,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(ReplacementTransform(lr_eq3, lr_eq4), run_time=0.8)
        self.wait(0.6)

        # Calculate new learning rate
        new_lr = 0.2 * learning_rate * sqrt_max_ab
        lr_eq5 = MathTex(
            r"\gamma_{\text{new}} &= 0.2 \times",
            f"{learning_rate} ",
            rf"\times {sqrt_max_ab:.3f} \\",
            rf"&= {new_lr:.5f}",
            font_size=28,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(ReplacementTransform(lr_eq4, lr_eq5), run_time=0.9)
        self.wait(0.8)

        lr_comparison = Text(
            f"γ: {learning_rate:.5f} → {new_lr:.5f} (adaptive adj)",
            font_size=22,
        ).set_color(GREY_B)
        lr_comparison.next_to(lr_eq5, DOWN, buff=0.3)

        self.play(FadeIn(lr_comparison, shift=0.1 * DOWN), run_time=0.7)
        self.wait(1.2)

        # Note about adaptive scaling
        adaptive_note = (
            Paragraph(
                "Dynamically adjusts step size",
                "based on optimization progress",
                font_size=18,
                alignment="center",
            )
            .set_color(GREY_B)
            .next_to(lr_comparison, DOWN, buff=0.15)
        )

        self.play(FadeIn(adaptive_note, shift=0.1 * DOWN), run_time=0.6)
        self.wait(1.2)

        # Clean up step 6
        self.play(
            FadeOut(lr_eq5, shift=0.2 * DOWN),
            FadeOut(lr_comparison),
            FadeOut(adaptive_note),
            FadeOut(lr_title),
            run_time=0.6,
        )
        self.wait(0.3)

        self.play(
            FadeOut(step6_box),
            panel_lines[5].animate.set_color(WHITE),
            run_time=0.6,
        )
        self.wait(0.5)

        #########################################################
        # STEP SEVEN - Parameter Update: θ = θ - γO_t
        #########################################################
        self.next_section("parameter_update", skip_animations=False)

        step7 = panel_lines[6]
        step7_box = RoundedRectangle(
            corner_radius=0.08,
            width=step7.width + 0.30,
            height=step7.height + 0.20,
        )
        step7_box.set_fill(opacity=0).set_stroke(TEAL_B, width=2.4)
        step7_box.move_to(step7)

        self.add(step7_box)
        self.play(FadeIn(step7_box, scale=0.98), step7.animate.set_color(TEAL_B), run_time=0.6)
        self.wait(0.3)

        update_title = Text("Step 7: Parameter Update", font_size=26).set_color(BLUE_E)
        update_title.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(update_title), run_time=0.6)
        self.wait(0.5)

        # Show update formula with explicit step notation
        update_eq1 = MathTex(
            r"\theta_2 = \theta_1 - \gamma O_t",
            font_size=32,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(Write(update_eq1), run_time=0.8)
        self.wait(0.8)

        # Step 1: Recall θ₁ (current position after weight decay from Step 5)
        update_eq2 = MathTex(
            rf"\theta_1 = \begin{{pmatrix}} {theta_after_decay[0]:.5f} \\ {theta_after_decay[1]:.5f} \end{{pmatrix}}",
            font_size=28,
        ).move_to(ORIGIN + RIGHT * 0.5)
        decay_reminder = Text("(after weight decay)", font_size=18).set_color(GREY_B)
        decay_reminder.next_to(update_eq2, DOWN, buff=0.15)

        self.play(
            ReplacementTransform(update_eq1, update_eq2),
            FadeIn(decay_reminder, shift=0.1 * DOWN),
            run_time=0.8,
        )
        self.wait(0.6)

        # Step 2: Recall γ (learning rate)
        update_eq3 = MathTex(
            rf"\gamma = {learning_rate}",
            font_size=30,
        ).move_to(ORIGIN + RIGHT * 0.5)
        self.play(
            ReplacementTransform(update_eq2, update_eq3), FadeOut(decay_reminder), run_time=0.8
        )
        self.wait(0.6)

        # Step 3: Recall O_t (the preconditioned update from step 4)
        update_eq4 = MathTex(
            rf"O_1 = \begin{{pmatrix}} {O1[0]:.2f} \\ {O1[1]:.2f} \end{{pmatrix}}",
            font_size=28,
        ).move_to(ORIGIN + RIGHT * 0.5)
        update_eq4[0][-14:].set_color(GREEN)

        self.play(ReplacementTransform(update_eq3, update_eq4), run_time=0.8)
        self.wait(0.6)

        # Step 4: Calculate γO_t
        gamma_O = learning_rate * O1
        update_eq5 = MathTex(
            rf"\gamma O_1 = {learning_rate} \times \begin{{pmatrix}} {O1[0]:.2f} \\ {O1[1]:.2f} \end{{pmatrix}} = \begin{{pmatrix}} {gamma_O[0]:.4f} \\ {gamma_O[1]:.4f} \end{{pmatrix}}",
            font_size=24,
        ).move_to(ORIGIN + RIGHT * 0.5)
        self.play(ReplacementTransform(update_eq4, update_eq5), run_time=0.9)
        self.wait(0.8)

        # Step 5: Show the full substitution
        theta2 = theta_after_decay - gamma_O
        update_eq6 = MathTex(
            r"\theta_2 &= \theta_1 - \gamma O_1 \\",
            rf"&= \begin{{pmatrix}} {theta_after_decay[0]:.5f} \\ {theta_after_decay[1]:.5f} \end{{pmatrix}} - \begin{{pmatrix}} {gamma_O[0]:.4f} \\ {gamma_O[1]:.4f} \end{{pmatrix}}",
            font_size=24,
        ).move_to(ORIGIN + RIGHT * 0.5)

        self.play(ReplacementTransform(update_eq5, update_eq6), run_time=0.9)
        self.wait(0.8)

        # Step 6: Show final result
        update_eq7 = MathTex(
            rf"\theta_2 = \begin{{pmatrix}} {theta2[0]:.2f} \\ {theta2[1]:.2f} \end{{pmatrix}}",
            font_size=30,
        ).move_to(ORIGIN + RIGHT * 0.5)
        self.play(ReplacementTransform(update_eq6, update_eq7), run_time=0.9)
        self.wait(0.8)

        # Draw the update step (from decayed position to new position)
        # Note: Weight decay shift is tiny (~0.0004) so visually imperceptible
        update_arrow = Arrow(
            start=local_axes.c2p(theta_after_decay[0], theta_after_decay[1]),
            end=local_axes.c2p(theta2[0], theta2[1]),
            buff=0,
            color=BLUE_B,
            stroke_width=4.5,
        )

        theta2_dot = Dot(local_axes.c2p(theta2[0], theta2[1]), radius=0.07, color=BLUE_E)
        theta2_label = (
            MathTex(r"\theta_2").scale(0.6).next_to(theta2_dot, DOWN + RIGHT * 0.5, buff=0.15)
        )
        theta2_sublabel = (
            MathTex(rf"({theta2[0]:.2f},\ {theta2[1]:.2f})")
            .scale(0.45)
            .set_color(BLUE_B)
            .next_to(theta2_label, DOWN, buff=0.08)
        )

        self.play(
            FadeIn(update_arrow, scale=0.9),
            run_time=0.8,
        )
        self.wait(0.5)

        self.play(
            FadeIn(theta2_dot, scale=1.2),
            FadeIn(theta2_label, shift=0.1 * DOWN),
            FadeIn(theta2_sublabel, shift=0.05 * DOWN),
            FadeOut(update_eq7, shift=0.2 * DOWN),
            FadeOut(update_title),
            run_time=1.0,
        )
        self.wait(1.5)

        # Final cleanup - fade out intermediate arrows, keep θ₁, θ₂, and contours
        self.play(
            FadeOut(o1_arrow),
            FadeOut(o1_label),
            FadeOut(update_arrow),
            run_time=0.8,
        )
        self.wait(0.5)

        # Clean up step 7 highlight
        self.play(
            FadeOut(step7_box),
            panel_lines[6].animate.set_color(WHITE),
            run_time=0.6,
        )
        self.wait(1.0)


# just for gif generation for my hero section
class MuonOverview3DSimple(ThreeDScene):
    def construct(self):
        self.next_section("surface_intro", skip_animations=False)

        axes, labels = make_3d_axes()
        surface = make_surface(axes)
        self.set_camera_orientation(phi=70 * DEGREES, theta=-450 * DEGREES, zoom=ZOOM)

        subtitle = (
            MathTex(r"z = 0.5\,\big[(x^4 - 16x^2 + 5x) + (y^4 - 16y^2 + 5y)\big]")
            .scale(0.6)
            .to_edge(DL, buff=0.3)
        )
        title = Tex("Styblinski–Tang Function (2D)").scale(0.8).next_to(subtitle, UP, buff=0.25)

        self.add_fixed_orientation_mobjects(labels)
        self.play(FadeIn(axes), FadeIn(labels))
        self.play(FadeIn(surface, shift=0.5 * IN), run_time=1.2)
        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(0.5)

        self.next_section("show_contours", skip_animations=False)
        contour_levels = [-70, -50, -30, -10, 10, 30, 50, 80]
        contour_colors = [
            RED,
            interpolate_color(RED, PURPLE_E, 0.3),
            PURPLE_E,
            interpolate_color(PURPLE_E, BLUE_E, 0.3),
            BLUE_E,
            interpolate_color(BLUE_E, TEAL_E, 0.3),
            TEAL_E,
            GREEN,
        ]

        contours_3d = VGroup()
        for c, color in zip(contour_levels, contour_colors):
            base = ImplicitFunction(
                lambda x, y, c=c: styblinski_tang_fn(x, y) - c,
                x_range=(X_RANGE[0], X_RANGE[1]),
                y_range=(Y_RANGE[0], Y_RANGE[1]),
                color=color,
                stroke_opacity=1.0,
                stroke_width=3.5,
            )
            # Lift curve onto the surface height z=c
            base.apply_function(lambda p, c=c: axes.c2p(p[0], p[1], c))
            contours_3d.add(base)

        self.play(LaggedStart(*[FadeIn(c) for c in contours_3d], lag_ratio=0.1), run_time=2.0)
        self.wait(0.5)

        # Section 3: Rotate camera 360°
        self.next_section("rotate_view", skip_animations=False)

        T = 6
        self.begin_ambient_camera_rotation(rate=TAU / T)
        self.wait(T)
        self.stop_ambient_camera_rotation()
        self.wait(0.3)

        # Section 4: Flatten to 2D projection
        self.next_section("flatten_to_2d", skip_animations=False)

        # Flatten the surface while still viewing from 3D angle
        flattened_surface = surface.copy()
        flattened_surface.apply_function(lambda p: np.array([p[0], p[1], 0.02 * p[2]]))
        flattened_surface.set_fill(opacity=0.75)
        flattened_surface.set_stroke(width=0.3, opacity=0.35)

        # project 3d contours onto 2d plane
        self.play(
            ReplacementTransform(surface, flattened_surface),
            FadeOut(labels),  # Hide z-axis label when going to 2D
            contours_3d.animate.apply_function(lambda p: axes.c2p(*axes.p2c(p)[:2], 0.0)),
            run_time=2.0,
        )
        surface = flattened_surface

        # Top-down camera
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, gamma=0, run_time=1.5)
        self.wait(0.4)
