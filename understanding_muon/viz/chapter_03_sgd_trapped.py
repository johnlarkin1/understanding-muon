from __future__ import annotations

import numpy as np
from manim import (
    BLUE_B,
    BOLD,
    DEGREES,
    DOWN,
    GREEN,
    GREEN_B,
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
    Axes,
    Dot,
    FadeIn,
    FadeOut,
    Scene,
    Star,
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

# Approximate location of the local minimum where SGD gets trapped
# This is in the positive quadrant near (2.746, 2.746)
LOCAL_MIN_APPROX = np.array([2.746, 2.746], dtype=float)


# ============================
# New helper functions
# ============================


def mark_minimum(
    axes: ThreeDAxes | Axes, position: np.ndarray, label: str, color, is_3d: bool = True
) -> VGroup:
    """
    Places a star marker at a minimum location with a text label.

    Args:
        axes: Either ThreeDAxes or 2D Axes
        position: [x, y] coordinates of the minimum
        label: Text to display
        color: Color for the marker and label
        is_3d: True for 3D scene, False for 2D contour scene

    Returns:
        VGroup containing the star marker and label
    """
    x, y = float(position[0]), float(position[1])

    if is_3d:
        # 3D scene: place marker on the surface
        z = styblinski_tang_fn(x, y)
        star = Star(n=5, outer_radius=0.2, color=color, fill_opacity=0.8)
        star.rotate(90 * DEGREES, axis=RIGHT)  # Lay flat on surface
        star.move_to(axes.c2p(x, y, z + 0.1))
        star.set_shade_in_3d(True)

        label_tex = Tex(label).scale(0.6).set_color(color)
        label_tex.move_to(axes.c2p(x, y, z + 0.8))
    else:
        # 2D contour scene: place marker on the plane
        star = Star(n=5, outer_radius=0.15, color=color, fill_opacity=0.8)
        star.move_to(axes.c2p(x, y))

        label_tex = Tex(label).scale(0.5).set_color(color)
        label_tex.next_to(star, UP, buff=0.15)

    return VGroup(star, label_tex)


# ============================================
# Scene 1: SGD Trapped on 3D Surface
# ============================================
class SGDTrapped3D(ThreeDScene):
    """
    Shows SGD starting from (3.8, 3.8) and getting trapped in a local minimum.
    Demonstrates the limitation of vanilla gradient descent in non-convex landscapes.
    """

    def construct(self):
        # Setup scene
        axes, axis_labels = make_3d_axes()
        surface = make_surface(axes)
        self.set_camera_orientation(phi=85 * DEGREES, theta=-50 * DEGREES, zoom=ZOOM)

        # Billboard axis labels and add surface
        self.add_fixed_orientation_mobjects(axis_labels)
        self.play(FadeIn(axes), FadeIn(axis_labels))
        self.play(FadeIn(surface, shift=0.5 * IN), run_time=1.2)

        # Title
        title = Tex("Standard Gradient Descent (SGD)").to_edge(UL).scale(0.8)
        subtitle = Tex("Starting at ", r"$(3.8, 3.8)$", " — will it find the global minimum?").scale(
            0.5
        )
        subtitle.next_to(title, DOWN, buff=0.2, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(1)

        # Mark the global minimum with a green star
        global_min_marker = mark_minimum(
            axes, GLOBAL_MIN_2D, "Global Min", GREEN, is_3d=True
        )
        self.add_fixed_orientation_mobjects(global_min_marker[1])  # Billboard the label
        self.play(FadeIn(global_min_marker))
        self.wait(0.5)

        # Animate the SGD path
        ball = ball_at(axes, SGD_PATH[0])
        trail = TracedPath(
            ball.get_center, stroke_width=4, stroke_color=YELLOW, stroke_opacity=0.8
        )

        arr = grad_arrow_3d(axes, SGD_PATH[0], scale_xy=0.6)
        self.add(trail, ball, arr)

        step_time = 0.2

        for i in range(1, len(SGD_PATH)):
            p = SGD_PATH[i]
            # Update arrow
            new_arr = grad_arrow_3d(axes, p, scale_xy=0.6)

            # Move ball to new position
            new_pos = axes.c2p(p[0], p[1], styblinski_tang_fn(p[0], p[1]) + ball.radius)
            self.play(
                ball.animate.move_to(new_pos),
                Transform(arr, new_arr),
                run_time=step_time,
                rate_func=smooth,
            )

        # Ball is now trapped - mark the local minimum
        self.wait(0.5)
        local_min_marker = mark_minimum(
            axes, LOCAL_MIN_APPROX, "Local Min\\\\(trapped!)", RED, is_3d=True
        )
        self.add_fixed_orientation_mobjects(local_min_marker[1])  # Billboard the label
        self.play(FadeIn(local_min_marker))
        self.wait(1)

        # Add annotation pointing out the problem
        trapped_note = (
            Text("SGD converged to a local minimum", weight=BOLD, color=RED_B)
            .scale(0.5)
            .to_edge(DOWN)
        )
        self.add_fixed_in_frame_mobjects(trapped_note)
        self.play(Write(trapped_note))
        self.wait(1)

        # Rotate camera to show both minima
        self.move_camera(phi=75 * DEGREES, theta=-30 * DEGREES, run_time=2.5)
        self.wait(0.5)

        # 180° rotation to see the valley structure
        T = 8
        self.begin_ambient_camera_rotation(rate=TAU / (2 * T))
        self.wait(T)
        self.stop_ambient_camera_rotation()

        # Final message
        final_note = (
            Tex("The global minimum is ", r"$\sim$", "4 units away!")
            .scale(0.6)
            .set_color_by_gradient(GREEN_B, TEAL_B)
        )
        final_note.next_to(trapped_note, UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(final_note)
        self.play(FadeIn(final_note))
        self.wait(2)


# ============================================
# Scene 2: SGD Trapped on 2D Contour Map
# ============================================
class SGDTrappedContour(Scene):
    """
    Shows the same SGD path on a 2D contour map for clarity.
    Makes it obvious that there's a local vs global minimum.
    """

    def construct(self):
        # Setup 2D axes and contour lines
        ax = Axes(x_range=X_RANGE, y_range=Y_RANGE, x_length=8, y_length=8, tips=False)
        ax_labels = VGroup(ax.get_x_axis_label(Tex("x")), ax.get_y_axis_label(Tex("y")))

        levels = (-70, -60, -50, -40, -30, -20, -10, 0, 20, 50, 100)
        contours = make_contours(list(levels))
        plot_group = VGroup(contours).move_to(ax.get_origin())

        # Title
        title = Tex("SGD on Contour Map (Top-Down View)").to_edge(UL).scale(0.8)
        subtitle = Tex("Topographic visualization of loss landscape").scale(0.5)
        subtitle.next_to(title, DOWN, buff=0.2, aligned_edge=LEFT)

        self.play(FadeIn(ax), FadeIn(ax_labels))
        self.play(FadeIn(plot_group, shift=0.25 * UP, lag_ratio=0.05))
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(0.5)

        # Mark starting point
        start_marker = Dot(ax.c2p(START[0], START[1]), color=YELLOW, radius=0.08)
        start_label = Tex("Start").scale(0.5).next_to(start_marker, UP + RIGHT, buff=0.1)
        self.play(FadeIn(start_marker), Write(start_label))
        self.wait(0.3)

        # Mark both minima BEFORE animating the path
        local_min_marker = mark_minimum(
            ax, LOCAL_MIN_APPROX, "Local Min", RED, is_3d=False
        )
        global_min_marker = mark_minimum(
            ax, GLOBAL_MIN_2D, "Global Min", GREEN, is_3d=False
        )

        self.play(FadeIn(local_min_marker), FadeIn(global_min_marker))
        self.wait(1)

        # Animate the SGD path
        dot = Dot(ax.c2p(SGD_PATH[0, 0], SGD_PATH[0, 1]), color=YELLOW, radius=0.06)
        trail = TracedPath(dot.get_center, stroke_color=YELLOW, stroke_width=4)
        self.add(dot, trail)

        step_time = 0.15
        for i in range(1, len(SGD_PATH)):
            p = SGD_PATH[i]
            self.play(
                dot.animate.move_to(ax.c2p(float(p[0]), float(p[1]))),
                run_time=step_time,
                rate_func=smooth,
            )

        self.wait(0.5)

        # Highlight that we're stuck
        stuck_note = Text("Gradient ≈ 0: Optimization stopped", weight=BOLD, color=RED_B).scale(
            0.5
        )
        stuck_note.to_edge(DOWN)
        self.play(Write(stuck_note))
        self.wait(1)

        # Draw an arrow pointing from local to global
        arrow_start = ax.c2p(LOCAL_MIN_APPROX[0] - 0.5, LOCAL_MIN_APPROX[1] - 0.5)
        arrow_end = ax.c2p(GLOBAL_MIN_2D[0] + 0.5, GLOBAL_MIN_2D[1] + 0.5)
        from manim import Arrow

        arrow = Arrow(arrow_start, arrow_end, color=GREEN_B, stroke_width=3, buff=0.3)
        arrow_label = Tex("Global min\\\\is here!").scale(0.4).set_color(GREEN_B)
        arrow_label.next_to(arrow.get_center(), LEFT, buff=0.2)

        self.play(FadeIn(arrow), Write(arrow_label))
        self.wait(2)

        # Fade out everything for clean end
        self.play(
            FadeOut(VGroup(ax, ax_labels, plot_group, title, subtitle)),
            FadeOut(VGroup(start_marker, start_label, dot, trail)),
            FadeOut(VGroup(local_min_marker, global_min_marker)),
            FadeOut(VGroup(stuck_note, arrow, arrow_label)),
        )
