"""
Chapter 2: Optimizer Visualizations

This module contains Manim scenes for visualizing different optimization algorithms
on the Styblinski-Tang function, demonstrating the difference between vanilla SGD
and SGD with momentum.
"""

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    Arrow,
    Axes,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Line,
    MathTex,
    Scene,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDScene,
    VGroup,
    Write,
    interpolate_color,
)
from manim.utils.color import BLUE, GOLD, GREEN, ORANGE, PURPLE, RED, YELLOW

# Custom color palette (matching chapter_01_intro.py patterns)
BLUE_B = interpolate_color(BLUE, WHITE, 0.3)
BLUE_C = interpolate_color(BLUE, WHITE, 0.5)
GOLD_B = interpolate_color(GOLD, WHITE, 0.3)
GOLD_C = interpolate_color(GOLD, WHITE, 0.5)
GREEN_B = interpolate_color(GREEN, WHITE, 0.3)
GREEN_C = interpolate_color(GREEN, WHITE, 0.5)
ORANGE_B = interpolate_color(ORANGE, WHITE, 0.3)
PURPLE_B = interpolate_color(PURPLE, WHITE, 0.3)
PURPLE_C = interpolate_color(PURPLE, WHITE, 0.5)
RED_B = interpolate_color(RED, WHITE, 0.3)
RED_C = interpolate_color(RED, WHITE, 0.5)
YELLOW_B = interpolate_color(YELLOW, WHITE, 0.3)


def styblinski_tang(x: float, y: float) -> float:
    """
    Compute the Styblinski-Tang function for 2D.

    f(x, y) = 0.5 * (x^4 - 16x^2 + 5x + y^4 - 16y^2 + 5y)

    Global minimum at (-2.903534, -2.903534) with f(x*) H -78.33
    """
    return 0.5 * (x**4 - 16 * x**2 + 5 * x + y**4 - 16 * y**2 + 5 * y)


def gradient_styblinski_tang(x: float, y: float) -> np.ndarray:
    """
    Compute the gradient of the Styblinski-Tang function.

    f = [df/dx, df/dy]
    df/dx = 0.5 * (4x^3 - 32x + 5)
    df/dy = 0.5 * (4y^3 - 32y + 5)
    """
    df_dx = 0.5 * (4 * x**3 - 32 * x + 5)
    df_dy = 0.5 * (4 * y**3 - 32 * y + 5)
    return np.array([df_dx, df_dy])


class StyblinkskiTangVisualization(ThreeDScene):
    """
    Scene 1: Visualize the Styblinski-Tang function in 3D and 2D.

    Part A: Show 3D surface with 360� rotation
    Part B: Transition to 2D contour plot
    """

    def construct(self):
        """Main construction method for the scene."""
        # Part A: 3D Visualization
        self.show_3d_surface()

        # Part B: 2D Contour Visualization
        self.show_2d_contour()

    def show_3d_surface(self):
        """Show the 3D surface with rotation."""
        # Title
        title = Text("The Styblinski-Tang Function", font_size=48, weight="BOLD")
        title.to_edge(UP, buff=0.3)

        # Function formula (top-left)
        formula = MathTex(r"f(x, y) = \frac{1}{2}\sum_{i} (x_i^4 - 16x_i^2 + 5x_i)", font_size=36)
        formula.to_edge(LEFT, buff=0.75).shift(UP * 3)

        # Show title and formula
        self.add_fixed_in_frame_mobjects(title, formula)
        self.play(Write(title), run_time=1)
        self.play(FadeIn(formula, shift=DOWN * 0.2), run_time=1)
        self.wait(0.5)

        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[-80, 40, 20],
            x_length=8,
            y_length=8,
            z_length=5,
        )

        # Create surface
        surface = Surface(
            lambda u, v: axes.c2p(u, v, styblinski_tang(u, v)),
            u_range=[-4, 4],
            v_range=[-4, 4],
            resolution=(50, 50),
            fill_opacity=0.8,
        )

        # Color the surface based on z-value
        # Map z-values to colors: BLUE (low/good) -> RED (high/bad)
        surface.set_fill_by_value(
            axes=axes,
            colors=[(BLUE, -80), (BLUE_B, -40), (GOLD_B, 0), (RED_B, 40)],
            axis=2,  # z-axis
        )

        # Set initial camera position
        self.set_camera_orientation(phi=65 * np.pi / 180, theta=-45 * np.pi / 180)

        # Animate surface creation
        self.play(Create(axes), run_time=1)
        self.play(Create(surface), run_time=2)
        self.wait(1)

        # 360� rotation (8-10 seconds)
        self.begin_ambient_camera_rotation(rate=0.7)  # ~8.5 seconds for 360�
        self.wait(9)
        self.stop_ambient_camera_rotation()

        # Add question text
        question = Text("How should we walk down this surface?", font_size=32, color=YELLOW_B)
        question.to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(question)
        self.play(FadeIn(question, shift=UP * 0.2), run_time=1)
        self.wait(2)

        # Store for cleanup
        self.surface_objects = VGroup(axes, surface, question)
        self.title = title
        self.formula = formula

    def show_2d_contour(self):
        """Transition to 2D contour plot."""
        # Fade out 3D elements
        self.play(FadeOut(self.surface_objects), FadeOut(self.formula), run_time=1)

        # Move camera back to 2D view
        self.move_camera(phi=0, theta=-90 * np.pi / 180, run_time=1.5)

        # Create 2D axes
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=10,
            y_length=10,
            tips=False,
        )
        axes.add_coordinates()

        # Axis labels
        x_label = MathTex("x", font_size=36).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("y", font_size=36).next_to(axes.y_axis, UP)

        # Create contour lines (sample at multiple levels)
        contours = VGroup()
        levels = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20]

        for level in levels:
            # Determine color based on level
            if level < -50:
                color = BLUE_B
            elif level < 0:
                color = BLUE_C
            elif level < 20:
                color = GOLD_C
            else:
                color = RED_C

            # Create points for this contour level
            points = []
            resolution = 100
            for i in range(resolution):
                for j in range(resolution):
                    x = -5 + 10 * i / resolution
                    y = -5 + 10 * j / resolution
                    z = styblinski_tang(x, y)

                    # If close to this level, add point
                    if abs(z - level) < 3:
                        points.append(axes.c2p(x, y))

            if points:
                # Create dots for contour (simplified representation)
                for i in range(0, len(points), 5):  # Subsample for performance
                    dot = Dot(points[i], radius=0.02, color=color, fill_opacity=0.6)
                    contours.add(dot)

        # Mark global minimum
        global_min = Dot(axes.c2p(-2.903534, -2.903534), radius=0.15, color=GREEN, fill_opacity=1)
        global_min_label = Text("Global Min", font_size=20, color=GREEN)
        global_min_label.next_to(global_min, DOWN + RIGHT, buff=0.2)

        # Mark some local minima
        local_minima = VGroup()
        local_positions = [(2.7, -2.9), (-2.9, 2.7), (2.7, 2.7)]
        for pos in local_positions:
            local_min = Dot(axes.c2p(pos[0], pos[1]), radius=0.1, color=YELLOW_B, fill_opacity=1)
            local_minima.add(local_min)

        # Update title for 2D view
        new_title = Text("2D Contour Plot", font_size=40, weight="BOLD")
        new_title.to_edge(UP, buff=0.3)

        # Animate 2D visualization
        self.add_fixed_in_frame_mobjects(new_title, axes, x_label, y_label)
        self.play(FadeOut(self.title), FadeIn(new_title), run_time=0.5)
        self.play(Create(axes), Write(x_label), Write(y_label), run_time=1)
        self.play(FadeIn(contours), run_time=2)
        self.play(GrowFromCenter(global_min), Write(global_min_label), run_time=1)
        self.play(FadeIn(local_minima), run_time=1)
        self.wait(3)


class SGDVisualization(Scene):
    """
    Scene 2: Visualize SGD getting stuck in local minimum.

    Shows side-by-side 3D surface and 2D contour plot with ball rolling
    down using vanilla SGD with multiple parameter sets.
    """

    def construct(self):
        """Main construction method."""
        # Title
        title = Text("Stochastic Gradient Descent (SGD)", font_size=40, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1)
        self.wait(0.5)

        # Analytically chosen starting point that falls into local minimum
        starting_point = np.array([0.0, 3.0])

        # Test multiple parameter sets
        parameter_sets = [
            {"lr": 0.01, "description": "Too slow"},
            {"lr": 0.1, "description": "Gets stuck"},
            {"lr": 0.5, "description": "Unstable"},
        ]

        for i, params in enumerate(parameter_sets):
            self.show_sgd_optimization(starting_point, params["lr"], params["description"], title)

            if i < len(parameter_sets) - 1:
                self.wait(1)

    def show_sgd_optimization(self, start_pos, learning_rate, description, title):
        """Run SGD optimization visualization with given parameters."""
        # Setup: Create 2D contour plot (simpler than side-by-side for now)
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=6,
            y_length=6,
            tips=False,
        ).shift(DOWN * 0.5)

        # Axis labels
        x_label = MathTex("x", font_size=28).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("y", font_size=28).next_to(axes.y_axis, UP)

        # Parameter display
        param_text = Text(f"� = {learning_rate}  ({description})", font_size=28, color=GOLD_B)
        param_text.next_to(title, DOWN, buff=0.3)

        # SGD update formula (top-left)
        formula = MathTex(r"x_{t+1} = x_t - \eta \nabla f(x_t)", font_size=32)
        formula.to_edge(LEFT, buff=0.75).shift(UP * 2.5)

        # Create simplified contour background
        contours = self.create_contour_background(axes)

        # Mark global minimum
        global_min = Dot(axes.c2p(-2.903534, -2.903534), radius=0.12, color=GREEN)

        # Create ball at starting position
        ball = Dot(axes.c2p(start_pos[0], start_pos[1]), radius=0.2, color=GOLD_B, fill_opacity=0.9)

        # Show setup
        self.play(FadeIn(param_text), FadeIn(formula, shift=DOWN * 0.2), run_time=0.5)
        self.play(Create(axes), Write(x_label), Write(y_label), FadeIn(contours), run_time=1)
        self.play(GrowFromCenter(global_min), GrowFromCenter(ball), run_time=0.7)
        self.wait(0.5)

        # Run SGD optimization
        position = start_pos.copy()
        trajectory = VGroup()
        max_steps = 30

        for step in range(max_steps):
            # Compute gradient
            grad = gradient_styblinski_tang(position[0], position[1])

            # Check if converged (gradient very small)
            if np.linalg.norm(grad) < 0.1:
                break

            # Create gradient arrow
            arrow_start = axes.c2p(position[0], position[1])
            grad_normalized = grad / (np.linalg.norm(grad) + 1e-6)
            arrow_end = axes.c2p(
                position[0] + 0.5 * grad_normalized[0], position[1] + 0.5 * grad_normalized[1]
            )
            grad_arrow = Arrow(
                arrow_start,
                arrow_end,
                color=GOLD,
                buff=0,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.3,
            )

            # Show gradient briefly
            if step % 3 == 0:  # Only show every 3rd gradient for clarity
                self.play(Create(grad_arrow), run_time=0.2)

            # Update position
            new_position = position - learning_rate * grad

            # Clip to domain
            new_position = np.clip(new_position, -5, 5)

            # Add trajectory line
            traj_line = Line(
                axes.c2p(position[0], position[1]),
                axes.c2p(new_position[0], new_position[1]),
                color=GOLD_C,
                stroke_width=2,
                stroke_opacity=0.5,
            )
            trajectory.add(traj_line)

            # Animate ball movement
            self.play(
                ball.animate.move_to(axes.c2p(new_position[0], new_position[1])),
                Create(traj_line),
                run_time=0.5,
            )

            # Remove gradient arrow
            if step % 3 == 0:
                self.play(FadeOut(grad_arrow), run_time=0.1)

            position = new_position

        # Check if stuck in local minimum or reached global
        final_loss = styblinski_tang(position[0], position[1])
        global_loss = styblinski_tang(-2.903534, -2.903534)

        if final_loss > global_loss + 10:  # Stuck in local minimum
            result_text = Text("Stuck in local minimum!", font_size=28, color=RED_B)
            result_circle = Circle(radius=0.4, color=RED, stroke_width=4).move_to(ball.get_center())
        else:
            result_text = Text("Reached global minimum!", font_size=28, color=GREEN_B)
            result_circle = Circle(radius=0.4, color=GREEN, stroke_width=4).move_to(
                ball.get_center()
            )

        result_text.to_edge(DOWN, buff=0.5)
        loss_text = Text(f"Final loss: {final_loss:.2f}", font_size=24, color=WHITE)
        loss_text.next_to(result_text, DOWN, buff=0.2)

        # Show result
        self.play(Create(result_circle), run_time=0.5)
        self.play(FadeIn(result_text), Write(loss_text), run_time=1)
        self.wait(2)

        # Cleanup
        self.play(
            FadeOut(
                VGroup(
                    axes,
                    x_label,
                    y_label,
                    contours,
                    global_min,
                    ball,
                    trajectory,
                    result_circle,
                    result_text,
                    loss_text,
                    param_text,
                    formula,
                )
            ),
            run_time=0.5,
        )

    def create_contour_background(self, axes):
        """Create simplified contour plot background."""
        contours = VGroup()

        # Create grid of dots colored by function value
        resolution = 30
        for i in range(resolution):
            for j in range(resolution):
                x = -5 + 10 * i / (resolution - 1)
                y = -5 + 10 * j / (resolution - 1)
                z = styblinski_tang(x, y)

                # Color based on z value
                if z < -50:
                    color = BLUE_B
                    opacity = 0.6
                elif z < -20:
                    color = BLUE_C
                    opacity = 0.4
                elif z < 0:
                    color = GOLD_C
                    opacity = 0.3
                else:
                    color = RED_C
                    opacity = 0.2

                dot = Dot(axes.c2p(x, y), radius=0.025, color=color, fill_opacity=opacity)
                contours.add(dot)

        return contours


class SGDMomentumVisualization(Scene):
    """
    Scene 3: Visualize SGD with momentum reaching global minimum.

    Same setup as Scene 2, but with momentum allowing escape from local minima.
    """

    def construct(self):
        """Main construction method."""
        # Title
        title = Text("SGD with Momentum", font_size=40, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1)
        self.wait(0.5)

        # Same starting point as Scene 2
        starting_point = np.array([0.0, 3.0])

        # Test multiple parameter sets
        parameter_sets = [
            {"lr": 0.01, "beta": 0.9, "description": "Slow but stable"},
            {"lr": 0.1, "beta": 0.9, "description": "Good balance"},
            {"lr": 0.1, "beta": 0.95, "description": "High momentum"},
        ]

        for i, params in enumerate(parameter_sets):
            self.show_momentum_optimization(
                starting_point, params["lr"], params["beta"], params["description"], title
            )

            if i < len(parameter_sets) - 1:
                self.wait(1)

    def show_momentum_optimization(self, start_pos, learning_rate, beta, description, title):
        """Run SGD with momentum optimization visualization."""
        # Setup: Create 2D contour plot
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=6,
            y_length=6,
            tips=False,
        ).shift(DOWN * 0.5)

        # Axis labels
        x_label = MathTex("x", font_size=28).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("y", font_size=28).next_to(axes.y_axis, UP)

        # Parameter display
        param_text = Text(
            f"� = {learning_rate}, � = {beta}  ({description})", font_size=28, color=PURPLE_B
        )
        param_text.next_to(title, DOWN, buff=0.3)

        # Momentum formulas (top-left)
        formulas = VGroup(
            MathTex(r"v_{t+1} = \beta v_t + \nabla f(x_t)", font_size=28),
            MathTex(r"x_{t+1} = x_t - \eta v_{t+1}", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        formulas.to_edge(LEFT, buff=0.75).shift(UP * 2.5)

        # Create simplified contour background
        contours = self.create_contour_background(axes)

        # Mark global minimum
        global_min = Dot(axes.c2p(-2.903534, -2.903534), radius=0.12, color=GREEN)

        # Create ball at starting position
        ball = Dot(
            axes.c2p(start_pos[0], start_pos[1]), radius=0.2, color=PURPLE_B, fill_opacity=0.9
        )

        # Show setup
        self.play(FadeIn(param_text), FadeIn(formulas, shift=DOWN * 0.2), run_time=0.5)
        self.play(Create(axes), Write(x_label), Write(y_label), FadeIn(contours), run_time=1)
        self.play(GrowFromCenter(global_min), GrowFromCenter(ball), run_time=0.7)
        self.wait(0.5)

        # Run SGD with momentum
        position = start_pos.copy()
        velocity = np.array([0.0, 0.0])
        trajectory = VGroup()
        max_steps = 40

        for step in range(max_steps):
            # Compute gradient
            grad = gradient_styblinski_tang(position[0], position[1])

            # Update velocity with momentum
            velocity = beta * velocity + grad

            # Check if converged
            if np.linalg.norm(velocity) < 0.05:
                break

            # Create gradient arrow (GOLD)
            arrow_start = axes.c2p(position[0], position[1])
            grad_normalized = grad / (np.linalg.norm(grad) + 1e-6)
            grad_arrow_end = axes.c2p(
                position[0] + 0.4 * grad_normalized[0], position[1] + 0.4 * grad_normalized[1]
            )
            grad_arrow = Arrow(
                arrow_start,
                grad_arrow_end,
                color=GOLD,
                buff=0,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.3,
            )

            # Create velocity arrow (PURPLE)
            vel_normalized = velocity / (np.linalg.norm(velocity) + 1e-6)
            vel_arrow_end = axes.c2p(
                position[0] + 0.6 * vel_normalized[0], position[1] + 0.6 * vel_normalized[1]
            )
            vel_arrow = Arrow(
                arrow_start,
                vel_arrow_end,
                color=PURPLE,
                buff=0,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.25,
            )

            # Show arrows briefly (every 4th step for clarity)
            if step % 4 == 0:
                self.play(Create(grad_arrow), run_time=0.15)
                self.play(Create(vel_arrow), run_time=0.15)

            # Update position
            new_position = position - learning_rate * velocity

            # Clip to domain
            new_position = np.clip(new_position, -5, 5)

            # Add trajectory line
            traj_line = Line(
                axes.c2p(position[0], position[1]),
                axes.c2p(new_position[0], new_position[1]),
                color=PURPLE_C,
                stroke_width=2,
                stroke_opacity=0.6,
            )
            trajectory.add(traj_line)

            # Animate ball movement
            self.play(
                ball.animate.move_to(axes.c2p(new_position[0], new_position[1])),
                Create(traj_line),
                run_time=0.5,
            )

            # Remove arrows
            if step % 4 == 0:
                self.play(FadeOut(grad_arrow), FadeOut(vel_arrow), run_time=0.1)

            position = new_position

        # Check result
        final_loss = styblinski_tang(position[0], position[1])
        global_loss = styblinski_tang(-2.903534, -2.903534)

        if final_loss > global_loss + 10:
            result_text = Text("Stuck in local minimum!", font_size=28, color=RED_B)
            result_circle = Circle(radius=0.4, color=RED, stroke_width=4).move_to(ball.get_center())
        else:
            result_text = Text("Global minimum reached!", font_size=28, color=GREEN_B)
            result_circle = Circle(radius=0.4, color=GREEN, stroke_width=4).move_to(
                ball.get_center()
            )

        result_text.to_edge(DOWN, buff=0.5)
        loss_text = Text(f"Final loss: {final_loss:.2f}", font_size=24, color=WHITE)
        loss_text.next_to(result_text, DOWN, buff=0.2)

        # Show result
        self.play(Create(result_circle), run_time=0.5)
        self.play(FadeIn(result_text), Write(loss_text), run_time=1)
        self.wait(2)

        # Cleanup
        self.play(
            FadeOut(
                VGroup(
                    axes,
                    x_label,
                    y_label,
                    contours,
                    global_min,
                    ball,
                    trajectory,
                    result_circle,
                    result_text,
                    loss_text,
                    param_text,
                    formulas,
                )
            ),
            run_time=0.5,
        )

    def create_contour_background(self, axes):
        """Create simplified contour plot background."""
        contours = VGroup()

        # Create grid of dots colored by function value
        resolution = 30
        for i in range(resolution):
            for j in range(resolution):
                x = -5 + 10 * i / (resolution - 1)
                y = -5 + 10 * j / (resolution - 1)
                z = styblinski_tang(x, y)

                # Color based on z value
                if z < -50:
                    color = BLUE_B
                    opacity = 0.6
                elif z < -20:
                    color = BLUE_C
                    opacity = 0.4
                elif z < 0:
                    color = GOLD_C
                    opacity = 0.3
                else:
                    color = RED_C
                    opacity = 0.2

                dot = Dot(axes.c2p(x, y), radius=0.025, color=color, fill_opacity=opacity)
                contours.add(dot)

        return contours
