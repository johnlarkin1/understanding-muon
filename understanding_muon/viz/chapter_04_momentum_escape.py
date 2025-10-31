from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    BLUE_B,
    BLUE_C,
    BOLD,
    DEGREES,
    DOWN,
    GREEN,
    IN,
    LEFT,
    RED,
    RED_B,
    RIGHT,
    TAU,
    TEAL_B,
    UL,
    UP,
    UR,
    YELLOW,
    YELLOW_B,
    Arrow3D,
    Axes,
    Dot,
    FadeIn,
    FadeOut,
    Scene,
    Tex,
    Text,
    ThreeDAxes,
    ThreeDScene,
    TracedPath,
    Transform,
    VGroup,
    Write,
    smooth,
)

# Import all utilities from chapter 02
from understanding_muon.viz.chapter_02_optimizers import (
    GLOBAL_MIN_2D,
    MOM_PATH,
    SGD_PATH,
    START,
    X_RANGE,
    Y_RANGE,
    ZOOM,
    ball_at,
    grad_arrow_3d,
    make_3d_axes,
    make_contours,
    make_surface,
    styblinski_tang_fn,
)

# Import from chapter 03
from understanding_muon.viz.chapter_03_sgd_trapped import LOCAL_MIN_APPROX, mark_minimum

# ============================
# New helper functions
# ============================


def momentum_velocity_arrow(
    axes: ThreeDAxes, p: np.ndarray, velocity: np.ndarray, scale: float = 0.5
) -> Arrow3D:
    """
    Arrow showing momentum/velocity direction (accumulated gradient history).

    Unlike gradient arrows (which show instantaneous descent direction),
    this shows the direction the ball is actually moving based on momentum.

    Args:
        axes: ThreeDAxes for coordinate mapping
        p: Current position [x, y]
        velocity: Current velocity vector [vx, vy]
        scale: Scaling factor for arrow length

    Returns:
        Arrow3D showing velocity direction
    """
    x, y = float(p[0]), float(p[1])
    v_norm = np.linalg.norm(velocity)

    if v_norm < 1e-9:
        # No velocity, return a tiny arrow
        start = axes.c2p(x, y, styblinski_tang_fn(x, y) + 0.25)
        end = axes.c2p(x, y, styblinski_tang_fn(x, y) + 0.3)
        return Arrow3D(start=start, end=end, color=BLUE_C, stroke_width=6)

    # Normalize velocity and scale it
    v_dir = velocity / v_norm
    dx, dy = scale * v_dir[0], scale * v_dir[1]

    # Compute z change along velocity direction
    dz = styblinski_tang_fn(x + dx, y + dy) - styblinski_tang_fn(x, y)

    start = axes.c2p(x, y, styblinski_tang_fn(x, y) + 0.25)
    end = axes.c2p(x + dx, y + dy, styblinski_tang_fn(x, y) + dz + 0.25)

    return Arrow3D(start=start, end=end, color=BLUE_C, stroke_width=6)


def compute_momentum_velocities(path: np.ndarray, lr: float = 0.05, beta: float = 0.8) -> list:
    """
    Reconstruct velocity vectors for each step of the momentum path.

    This recomputes what the velocity was at each step, since the path
    only contains positions, not velocities.

    Returns:
        List of velocity vectors, one per step
    """
    from understanding_muon.viz.chapter_02_optimizers import styblinski_tang_grad

    velocities = []
    v = np.zeros(2, dtype=float)

    for i in range(len(path)):
        x, y = path[i]
        g = styblinski_tang_grad(x, y)
        v = beta * v - lr * g
        velocities.append(v.copy())

    return velocities


# ============================================
# Scene 1: Momentum Escape on 3D Surface
# ============================================
class MomentumEscape3D(ThreeDScene):
    """
    Side-by-side comparison: SGD (yellow) gets trapped, Momentum (blue) escapes.
    Shows both balls starting from the same position and taking different paths.
    """

    def construct(self):
        # Setup scene
        axes, axis_labels = make_3d_axes()
        surface = make_surface(axes)
        self.set_camera_orientation(phi=55 * DEGREES, theta=-50 * DEGREES, zoom=ZOOM)

        # Billboard axis labels and add surface
        self.add_fixed_orientation_mobjects(axis_labels)
        self.play(FadeIn(axes), FadeIn(axis_labels))
        self.play(FadeIn(surface, shift=0.5 * IN), run_time=1.2)

        # Title
        title = Tex("SGD vs SGD with Momentum").to_edge(UL).scale(0.8)
        subtitle = VGroup(
            Tex("Yellow = Vanilla SGD", color=YELLOW).scale(0.5),
            Tex("Blue = SGD + Momentum", color=BLUE).scale(0.5),
        ).arrange(RIGHT, buff=0.5)
        subtitle.next_to(title, DOWN, buff=0.2, aligned_edge=LEFT)

        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(1)

        # Mark the global minimum with green star
        global_min_marker = mark_minimum(axes, GLOBAL_MIN_2D, "Global Min", GREEN, is_3d=True)
        self.add_fixed_orientation_mobjects(global_min_marker[1])
        self.play(FadeIn(global_min_marker))
        self.wait(0.5)

        # Create two balls at the starting position
        ball_sgd = ball_at(axes, SGD_PATH[0], radius=0.15, color=YELLOW)
        ball_mom = ball_at(axes, MOM_PATH[0], radius=0.15, color=BLUE)

        trail_sgd = TracedPath(
            ball_sgd.get_center, stroke_width=4, stroke_color=YELLOW, stroke_opacity=0.8
        )
        trail_mom = TracedPath(
            ball_mom.get_center, stroke_width=4, stroke_color=BLUE, stroke_opacity=0.8
        )

        # Gradient arrows for both
        arr_sgd = grad_arrow_3d(axes, SGD_PATH[0], scale_xy=0.5)
        arr_sgd.set_color(YELLOW_B)

        # For momentum, we'll show velocity arrows instead of gradient
        mom_velocities = compute_momentum_velocities(MOM_PATH)
        arr_mom = momentum_velocity_arrow(axes, MOM_PATH[0], mom_velocities[0], scale=0.5)

        self.add(trail_sgd, trail_mom, ball_sgd, ball_mom, arr_sgd, arr_mom)
        self.wait(0.5)

        # Animate both paths, but SGD stops when it reaches its end
        step_time = 0.2
        max_steps = max(len(SGD_PATH), len(MOM_PATH))

        for i in range(1, max_steps):
            animations = []

            # SGD ball (only if not finished)
            if i < len(SGD_PATH):
                p_sgd = SGD_PATH[i]
                new_pos_sgd = axes.c2p(
                    p_sgd[0], p_sgd[1], styblinski_tang_fn(p_sgd[0], p_sgd[1]) + ball_sgd.radius
                )
                animations.append(ball_sgd.animate.move_to(new_pos_sgd))

                # Update SGD gradient arrow
                new_arr_sgd = grad_arrow_3d(axes, p_sgd, scale_xy=0.5)
                new_arr_sgd.set_color(YELLOW_B)
                animations.append(Transform(arr_sgd, new_arr_sgd))

            # Momentum ball (always continues)
            if i < len(MOM_PATH):
                p_mom = MOM_PATH[i]
                new_pos_mom = axes.c2p(
                    p_mom[0], p_mom[1], styblinski_tang_fn(p_mom[0], p_mom[1]) + ball_mom.radius
                )
                animations.append(ball_mom.animate.move_to(new_pos_mom))

                # Update momentum velocity arrow
                new_arr_mom = momentum_velocity_arrow(axes, p_mom, mom_velocities[i], scale=0.5)
                animations.append(Transform(arr_mom, new_arr_mom))

            if animations:
                self.play(*animations, run_time=step_time, rate_func=smooth)

        self.wait(1)

        # Mark where SGD stopped (local minimum)
        local_min_marker = mark_minimum(
            axes, LOCAL_MIN_APPROX, "Local Min\\\\(SGD trapped)", RED, is_3d=True
        )
        self.add_fixed_orientation_mobjects(local_min_marker[1])
        self.play(FadeIn(local_min_marker))
        self.wait(1)

        # Add success/failure labels
        sgd_label = Text("SGD: Trapped ❌", weight=BOLD, color=YELLOW_B).scale(0.4)
        mom_label = Text("Momentum: Success ✓", weight=BOLD, color=BLUE_B).scale(0.4)

        label_group = VGroup(sgd_label, mom_label).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        label_group.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(label_group)
        self.play(Write(label_group))
        self.wait(1)

        # Rotate camera to show full landscape
        self.move_camera(phi=75 * DEGREES, theta=-30 * DEGREES, run_time=2.5)
        self.wait(0.5)

        # 360° rotation to appreciate the paths
        T = 10
        self.begin_ambient_camera_rotation(rate=TAU / T)
        self.wait(T)
        self.stop_ambient_camera_rotation()

        # Final insight
        insight = (
            Tex("Momentum carries the ball over the local minimum!")
            .scale(0.6)
            .set_color_by_gradient(BLUE_B, TEAL_B)
        )
        insight.next_to(label_group, UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(insight)
        self.play(FadeIn(insight))
        self.wait(2)


# ============================================
# Scene 2: Momentum Escape on 2D Contours
# ============================================
class MomentumEscapeContour(Scene):
    """
    2D contour map showing both SGD and Momentum paths side-by-side.
    Clearly shows momentum escaping the local minimum basin.
    """

    def construct(self):
        # Setup 2D axes and contour lines
        ax = Axes(x_range=X_RANGE, y_range=Y_RANGE, x_length=8, y_length=8, tips=False)
        ax_labels = VGroup(ax.get_x_axis_label(Tex("x")), ax.get_y_axis_label(Tex("y")))

        levels = (-70, -60, -50, -40, -30, -20, -10, 0, 20, 50, 100)
        contours = make_contours(list(levels))
        plot_group = VGroup(contours).move_to(ax.get_origin())

        # Title
        title = Tex("SGD vs Momentum: Contour View").to_edge(UL).scale(0.8)
        legend = VGroup(
            Tex("Yellow = SGD (trapped)", color=YELLOW).scale(0.5),
            Tex("Blue = Momentum (escapes)", color=BLUE).scale(0.5),
        ).arrange(RIGHT, buff=0.5)
        legend.next_to(title, DOWN, buff=0.2, aligned_edge=LEFT)

        self.play(FadeIn(ax), FadeIn(ax_labels))
        self.play(FadeIn(plot_group, shift=0.25 * UP, lag_ratio=0.05))
        self.play(FadeIn(title), FadeIn(legend))
        self.wait(0.5)

        # Mark starting point and both minima
        start_marker = Dot(ax.c2p(START[0], START[1]), color=YELLOW, radius=0.08)
        start_label = Tex("Start").scale(0.5).next_to(start_marker, UP + RIGHT, buff=0.1)
        self.play(FadeIn(start_marker), Write(start_label))
        self.wait(0.3)

        local_min_marker = mark_minimum(ax, LOCAL_MIN_APPROX, "Local", RED, is_3d=False)
        global_min_marker = mark_minimum(ax, GLOBAL_MIN_2D, "Global", GREEN, is_3d=False)
        self.play(FadeIn(local_min_marker), FadeIn(global_min_marker))
        self.wait(1)

        # Create two dots for the two paths
        dot_sgd = Dot(ax.c2p(SGD_PATH[0, 0], SGD_PATH[0, 1]), color=YELLOW, radius=0.06)
        dot_mom = Dot(ax.c2p(MOM_PATH[0, 0], MOM_PATH[0, 1]), color=BLUE, radius=0.06)

        trail_sgd = TracedPath(dot_sgd.get_center, stroke_color=YELLOW, stroke_width=4)
        trail_mom = TracedPath(dot_mom.get_center, stroke_color=BLUE, stroke_width=4)

        self.add(dot_sgd, dot_mom, trail_sgd, trail_mom)
        self.wait(0.3)

        # Animate both paths simultaneously
        step_time = 0.15
        max_steps = max(len(SGD_PATH), len(MOM_PATH))

        for i in range(1, max_steps):
            animations = []

            # SGD dot (only if not finished)
            if i < len(SGD_PATH):
                p_sgd = SGD_PATH[i]
                animations.append(dot_sgd.animate.move_to(ax.c2p(float(p_sgd[0]), float(p_sgd[1]))))

            # Momentum dot (always continues)
            if i < len(MOM_PATH):
                p_mom = MOM_PATH[i]
                animations.append(dot_mom.animate.move_to(ax.c2p(float(p_mom[0]), float(p_mom[1]))))

            if animations:
                self.play(*animations, run_time=step_time, rate_func=smooth)

        self.wait(1)

        # Annotate the key insight
        insight1 = Text("SGD stops at local minimum", weight=BOLD, color=YELLOW_B).scale(0.45)
        insight2 = Text("Momentum escapes to global minimum", weight=BOLD, color=BLUE_B).scale(0.45)
        insights = VGroup(insight1, insight2).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        insights.to_edge(DOWN)
        self.play(Write(insights))
        self.wait(1.5)

        # Draw arrows showing the escape
        from manim import CurvedArrow

        # Show how momentum "jumped over" the barrier
        escape_start = ax.c2p(LOCAL_MIN_APPROX[0] - 0.8, LOCAL_MIN_APPROX[1] - 0.8)
        escape_end = ax.c2p(GLOBAL_MIN_2D[0] + 0.8, GLOBAL_MIN_2D[1] + 0.8)
        escape_arrow = CurvedArrow(
            escape_start, escape_end, color=BLUE_B, stroke_width=3, angle=-TAU / 6
        )
        escape_label = Tex("Momentum\\\\carries over!").scale(0.4).set_color(BLUE_B)
        escape_label.next_to(escape_arrow.get_center(), LEFT + UP, buff=0.1)

        self.play(FadeIn(escape_arrow), Write(escape_label))
        self.wait(2)

        # Clean fade out
        all_objects = VGroup(
            ax,
            ax_labels,
            plot_group,
            title,
            legend,
            start_marker,
            start_label,
            local_min_marker,
            global_min_marker,
            dot_sgd,
            dot_mom,
            trail_sgd,
            trail_mom,
            insights,
            escape_arrow,
            escape_label,
        )
        self.play(FadeOut(all_objects))


# ============================================
# Scene 3 (BONUS): Velocity Vector Comparison
# ============================================
class VelocityVectorDemo(ThreeDScene):
    """
    Educational scene showing the difference between gradient vectors
    and momentum/velocity vectors at a few key points along the momentum path.
    """

    def construct(self):
        # Setup scene
        axes, axis_labels = make_3d_axes()
        surface = make_surface(axes)
        self.set_camera_orientation(phi=80 * DEGREES, theta=-45 * DEGREES, zoom=ZOOM)

        self.add_fixed_orientation_mobjects(axis_labels)
        self.play(FadeIn(axes), FadeIn(axis_labels), FadeIn(surface))

        # Title
        title = Tex("Gradient vs Velocity Vectors").to_edge(UL).scale(0.8)
        subtitle = VGroup(
            Tex("Red = Gradient (instantaneous)", color=RED).scale(0.45),
            Tex("Blue = Velocity (accumulated)", color=BLUE).scale(0.45),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        subtitle.next_to(title, DOWN, buff=0.2, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(1)

        # Pick a few interesting points along the momentum path
        mom_velocities = compute_momentum_velocities(MOM_PATH)
        demo_indices = [0, 10, 20, 30, 40]  # Sample points

        for idx in demo_indices:
            if idx >= len(MOM_PATH):
                continue

            p = MOM_PATH[idx]
            v = mom_velocities[idx]

            # Place ball at this position
            ball = ball_at(axes, p, radius=0.12, color=BLUE_B)

            # Gradient arrow (red)
            grad_arr = grad_arrow_3d(axes, p, scale_xy=0.6)
            grad_arr.set_color(RED_B)

            # Velocity arrow (blue)
            vel_arr = momentum_velocity_arrow(axes, p, v, scale=0.6)
            vel_arr.set_color(BLUE_B)

            # Add position label
            pos_label = (
                Tex(rf"Step {idx}: $({p[0]:.2f}, {p[1]:.2f})$")
                .scale(0.5)
                .to_corner(UR)
                .shift(DOWN * 0.5)
            )
            self.add_fixed_in_frame_mobjects(pos_label)

            self.play(FadeIn(ball), FadeIn(grad_arr), FadeIn(vel_arr), FadeIn(pos_label))
            self.wait(1.5)

            # Rotate camera slightly to see vectors
            if idx < demo_indices[-1]:
                self.play(
                    FadeOut(ball),
                    FadeOut(grad_arr),
                    FadeOut(vel_arr),
                    FadeOut(pos_label),
                    run_time=0.3,
                )

        # Final message
        final_note = (
            Tex("Velocity accumulates history,\\\\giving momentum to escape local minima")
            .scale(0.5)
            .set_color(BLUE_B)
            .to_edge(DOWN)
        )
        self.add_fixed_in_frame_mobjects(final_note)
        self.play(Write(final_note))
        self.wait(3)
