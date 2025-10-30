"""
Rosenbrock (banana) valley + SGD multi-step trajectory (3D)

Scene goal:
- Visualize the Rosenbrock function f(x, y) = (1-x)^2 + 10 (y - x^2)^2 as a 3D surface
- Show a single point (x0, y0) on the surface with tangent plane
- Draw ∇f(x0, y0) (uphill) and -∇f(x0, y0) (steepest descent) as true 3D vectors
- Highlight focus region with bounding box and zoom in
- Show multi-step SGD trajectory (4-5 steps) with learning rate η=0.02
- Add the valley floor curve y = x^2 (z = (1 - x)^2) to highlight curvature
- Display running annotations (step counter, loss value)

Usage:
    uv run manim -pql understanding_muon/viz/chapter_02_optimizers.py OptimizerPowerRosenbrock
    uv run manim -pqh understanding_muon/viz/chapter_02_optimizers.py OptimizerPowerRosenbrock
"""

import numpy as np
from manim import (
    BLUE,
    BLUE_E,
    DARK_GREY,
    DEGREES,
    DOWN,
    GRAY,
    GREEN,
    GREY,
    ORANGE,
    RED,
    RIGHT,
    UL,
    UP,
    UR,
    YELLOW,
    YELLOW_E,
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    MathTex,
    ParametricFunction,
    Sphere,
    Text,
    ThreeDAxes,
    ThreeDScene,
    VGroup,
    Write,
    interpolate_color,
)

# --- Optional imports for 3D arrows/dots with graceful fallbacks ---
try:  # Arrow3D exists in Manim Community (>=0.15). Fall back to Line if missing.
    from manim import Arrow3D  # type: ignore
except Exception:
    Arrow3D = None  # fallback to Line if not available

try:
    from manim import Dot3D  # type: ignore
except Exception:
    Dot3D = None  # fallback to Sphere if not available


# --- Color variants to match your chapter_01 style ---
def lighten(color, amount=0.35):
    # simple linear interpolation toward white
    from manim import WHITE, interpolate_color

    return interpolate_color(color, WHITE, amount)


BLUE_B = lighten(BLUE, 0.30)
BLUE_C = lighten(BLUE, 0.50)
GREEN_B = lighten(GREEN, 0.30)
GREEN_C = lighten(GREEN, 0.50)
RED_B = lighten(RED, 0.30)
RED_C = lighten(RED, 0.50)
GOLD = YELLOW_E
GOLD_B = lighten(GOLD, 0.30)
GOLD_C = lighten(GOLD, 0.50)
DARK_GRAY = DARK_GREY


# --- Utility: robust arrow/dot creators that work across Manim versions ---
def arrow3d(start, end, color=YELLOW, stroke_width=6):
    if Arrow3D is not None:
        return Arrow3D(start=start, end=end, color=color, stroke_width=stroke_width)
    # Fallback: plain Line in 3D
    return Line(start, end, color=color, stroke_width=stroke_width)


def dot3d(point, color=YELLOW, radius=0.05):
    if Dot3D is not None:
        return Dot3D(point=point, color=color, radius=radius)
    # Fallback: small sphere
    s = Sphere(radius=radius, color=color)
    s.move_to(point)
    return s


class OptimizerPowerRosenbrock(ThreeDScene):
    def rosenbrock(self, x, y, a=1.0, b=10.0):
        """Rosenbrock function with b=10 for gentler surface"""
        return (a - x) ** 2 + b * (y - x**2) ** 2

    def grad_rosenbrock(self, x, y, a=1.0, b=10.0):
        """Gradient of Rosenbrock function"""
        # ∂f/∂x = 2(x - a) - 4 b x (y - x^2)
        # ∂f/∂y = 2 b (y - x^2)
        dfx = 2 * (x - a) - 4 * b * x * (y - x**2)
        dfy = 2 * b * (y - x**2)
        return np.array([dfx, dfy])

    def construct(self):
        # --- 1) Title (fixed on screen) ---
        title = Text("Optimizers on the Rosenbrock Valley", font_size=44, weight="BOLD")
        subtitle = Text("Gradient vs. SGD proposal at a single point", font_size=26, color=GREY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.2)
        title_group.to_edge(UP, buff=0.6)
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.2))
        self.wait(0.5)

        # --- 2) Axes & Camera ---
        # With b=10, z_range is much gentler (0 to ~40 instead of 0 to ~80)
        axes = ThreeDAxes(
            x_range=[-2.2, 2.2, 1],
            y_range=[-0.8, 3.0, 1],
            z_range=[0.0, 40.0, 10],
            x_length=7,
            y_length=7,
            z_length=4.5,
        )
        axes_labels = VGroup(
            MathTex("x").scale(0.9).next_to(axes.x_axis.get_end(), RIGHT, buff=0.1),
            MathTex("y").scale(0.9).next_to(axes.y_axis.get_end(), UP, buff=0.1),
            MathTex("f(x,y)").scale(0.9).next_to(axes.z_axis.get_end(), UP, buff=0.1),
        )
        self.play(FadeIn(axes), FadeIn(axes_labels), run_time=0.8)

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES, zoom=1.0)
        self.wait(0.25)

        # --- 3) Rosenbrock Surface ---
        # Note: To maximize compatibility across Manim versions, we build the surface with ParametricFunction of a grid.
        # But Manim's Surface/ParametricSurface is typical. Here, we use a lightweight mesh of small lines for stability.
        # If your Manim has Surface, uncomment the block below for a smooth parametric surface.

        use_surface = True
        surface_mobject = None

        if use_surface:
            try:
                # Prefer the higher-level Surface if available
                from manim import Surface  # type: ignore

                def surf(u, v):
                    z = self.rosenbrock(u, v)
                    return axes.c2p(u, v, z)

                surface_mobject = Surface(
                    lambda u, v: surf(u, v),
                    u_range=[-2.0, 2.0],
                    v_range=[-0.5, 2.5],
                    resolution=(44, 44),
                    fill_opacity=0.87,
                    checkerboard_colors=[lighten(BLUE_E, 0.35), lighten(BLUE_E, 0.10)],
                    stroke_color=GRAY,
                    stroke_opacity=0.25,
                    stroke_width=0.5,
                )
                self.play(FadeIn(surface_mobject), run_time=1.2)
            except Exception:
                use_surface = False

        if not use_surface:
            # Fallback mesh (robust across versions): draw a grid of curves
            u_vals = np.linspace(-2.0, 2.0, 40)
            v_vals = np.linspace(-0.5, 2.5, 40)
            iso_u = VGroup(
                *[
                    ParametricFunction(
                        lambda t, uu=uu: axes.c2p(uu, t, self.rosenbrock(uu, t)),
                        t_range=[-0.5, 2.5],
                        color=lighten(BLUE_E, 0.35),
                        stroke_width=1.2,
                        stroke_opacity=0.7,
                    )
                    for uu in u_vals
                ]
            )
            iso_v = VGroup(
                *[
                    ParametricFunction(
                        lambda t, vv=vv: axes.c2p(t, vv, self.rosenbrock(t, vv)),
                        t_range=[-2.0, 2.0],
                        color=lighten(BLUE_E, 0.15),
                        stroke_width=1.2,
                        stroke_opacity=0.7,
                    )
                    for vv in v_vals
                ]
            )
            surface_mobject = VGroup(iso_u, iso_v)
            self.play(LaggedStart(Create(iso_u), Create(iso_v), lag_ratio=0.1), run_time=1.4)

        # --- 4) Valley Floor Curve (y = x^2, z = (1 - x)^2) ---
        valley = ParametricFunction(
            lambda t: axes.c2p(t, t**2, (1.0 - t) ** 2),
            t_range=[-2.0, 2.0],
            color=BLUE_C,
            stroke_width=4.0,
        )
        valley_label = Text("Valley floor (y = x^2)", font_size=20, color=BLUE_C)
        valley_label.add_updater(
            lambda m: m.move_to(axes.c2p(0.8, 0.64, (1 - 0.8) ** 2) + np.array([0.6, 0.3, 0]))
        )
        self.play(Create(valley), FadeIn(valley_label), run_time=1.0)

        # --- 5) Pick a single point & compute gradient ---
        x0, y0 = -1.2, 1.0
        z0 = self.rosenbrock(x0, y0)
        g = self.grad_rosenbrock(x0, y0)  # ∇f
        g_norm = np.linalg.norm(g)
        g_hat = g / g_norm

        p0 = axes.c2p(x0, y0, z0)
        pt = dot3d(p0, color=YELLOW, radius=0.055)
        self.play(FadeIn(pt), run_time=0.6)

        # Fixed-in-frame annotations (top-left) - updated formula with b=10
        f_eq = MathTex(r"f(x,y)=(1-x)^2 + 10\,(y-x^2)^2").scale(0.8)
        grad_eq = MathTex(r"\nabla f(x,y)=\big(2(x-1)-40x(y-x^2),\ 20(y-x^2)\big)").scale(0.7)
        numeric_vals = MathTex(
            rf"f(x_0,y_0)\!=\!{z0:.2f},\ \ \nabla f(x_0,y_0)\!=\!({g[0]:.1f},\,{g[1]:.1f})"
        ).scale(0.8)

        info_panel = VGroup(f_eq, grad_eq, numeric_vals).arrange(DOWN, buff=0.15)
        info_panel.to_corner(UL, buff=0.6)
        self.add_fixed_in_frame_mobjects(info_panel)
        self.play(FadeIn(info_panel), run_time=0.6)

        # --- 5.5) Add Tangent Plane at (x0, y0) ---
        # Tangent plane: z = z0 + grad[0]*(x-x0) + grad[1]*(y-y0)
        try:
            from manim import Surface  # type: ignore

            def tangent_surf(u, v):
                # u, v are offsets from (x0, y0)
                x_tangent = x0 + u
                y_tangent = y0 + v
                z_tangent = z0 + g[0] * u + g[1] * v
                return axes.c2p(x_tangent, y_tangent, z_tangent)

            tangent_plane = Surface(
                lambda u, v: tangent_surf(u, v),
                u_range=[-0.4, 0.4],
                v_range=[-0.4, 0.4],
                resolution=(12, 12),
                fill_opacity=0.6,
                fill_color=YELLOW,
                stroke_width=0.5,
            )
            tangent_label = Text("Tangent plane", font_size=18, color=YELLOW)
            tangent_label.add_updater(lambda m: m.move_to(p0 + np.array([0.8, 0.8, 0.5])))

            self.play(FadeIn(tangent_plane), FadeIn(tangent_label), run_time=0.8)
            self.wait(0.5)
        except Exception:
            tangent_plane = None
            tangent_label = None

        # --- 6) True 3D Gradient Vectors (follow surface curvature) ---
        # Calculate actual z-values at gradient endpoints for true 3D visualization
        scale = 0.5  # visibility scaling for gradient arrows

        # Uphill gradient (positive direction)
        x_grad_up = x0 + g_hat[0] * scale
        y_grad_up = y0 + g_hat[1] * scale
        z_grad_up = self.rosenbrock(x_grad_up, y_grad_up)  # Calculate actual z on surface
        p_grad_end = axes.c2p(x_grad_up, y_grad_up, z_grad_up)

        # Downhill gradient (negative direction)
        x_grad_down = x0 - g_hat[0] * scale
        y_grad_down = y0 - g_hat[1] * scale
        z_grad_down = self.rosenbrock(x_grad_down, y_grad_down)  # Calculate actual z on surface
        p_desc_end = axes.c2p(x_grad_down, y_grad_down, z_grad_down)

        grad_arrow = arrow3d(p0, p_grad_end, color=RED_B, stroke_width=8)
        desc_arrow = arrow3d(p0, p_desc_end, color=GREEN_B, stroke_width=8)

        grad_label = MathTex(r"\nabla f", color=RED_C).scale(0.9)
        desc_label = MathTex(r"-\nabla f", color=GREEN_C).scale(0.9)
        grad_label.add_updater(
            lambda m: m.move_to((p0 + p_grad_end) / 2 + np.array([0.0, 0.0, 0.6]))
        )
        desc_label.add_updater(
            lambda m: m.move_to((p0 + p_desc_end) / 2 + np.array([0.0, 0.0, 0.6]))
        )

        self.play(
            Create(grad_arrow),
            Create(desc_arrow),
            FadeIn(grad_label),
            FadeIn(desc_label),
            run_time=1.0,
        )
        self.wait(0.5)

        # Enhanced camera rotation during gradient demo (faster for engagement)
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(6.0)
        self.stop_ambient_camera_rotation()

        # Fade out tangent plane and gradient arrows before zooming
        fade_items = [grad_arrow, desc_arrow, grad_label, desc_label]
        if tangent_plane:
            fade_items.extend([tangent_plane, tangent_label])
        self.play(*[FadeOut(item) for item in fade_items], run_time=0.8)

        # --- 7) Add Bounding Box for Focus Region ---
        # Define focus region for zoom (where SGD trajectory will occur)
        x_min, x_max = -1.0, 1.5
        y_min, y_max = 0.0, 2.0
        z_avg = 5.0  # Approximate average z in focus region

        # Create wireframe bounding box
        corners = [
            axes.c2p(x_min, y_min, 0),
            axes.c2p(x_max, y_min, 0),
            axes.c2p(x_max, y_max, 0),
            axes.c2p(x_min, y_max, 0),
            axes.c2p(x_min, y_min, 15),
            axes.c2p(x_max, y_min, 15),
            axes.c2p(x_max, y_max, 15),
            axes.c2p(x_min, y_max, 15),
        ]

        # Bottom rectangle
        bbox_lines = VGroup(
            Line(corners[0], corners[1], color=ORANGE, stroke_width=3),
            Line(corners[1], corners[2], color=ORANGE, stroke_width=3),
            Line(corners[2], corners[3], color=ORANGE, stroke_width=3),
            Line(corners[3], corners[0], color=ORANGE, stroke_width=3),
            # Top rectangle
            Line(corners[4], corners[5], color=ORANGE, stroke_width=3),
            Line(corners[5], corners[6], color=ORANGE, stroke_width=3),
            Line(corners[6], corners[7], color=ORANGE, stroke_width=3),
            Line(corners[7], corners[4], color=ORANGE, stroke_width=3),
            # Vertical edges
            Line(corners[0], corners[4], color=ORANGE, stroke_width=3),
            Line(corners[1], corners[5], color=ORANGE, stroke_width=3),
            Line(corners[2], corners[6], color=ORANGE, stroke_width=3),
            Line(corners[3], corners[7], color=ORANGE, stroke_width=3),
        )

        bbox_label = Text("Focus region", font_size=22, color=ORANGE)
        bbox_label.add_updater(
            lambda m: m.move_to(axes.c2p((x_min + x_max) / 2, y_max, 15) + np.array([0, 0.3, 0]))
        )

        self.play(Create(bbox_lines), FadeIn(bbox_label), run_time=1.0)
        self.wait(0.5)

        # --- 8) Camera Zoom into Focus Region ---
        # For ThreeDScene, use move_camera() instead of self.camera.frame
        focus_point = axes.c2p((x_min + x_max) / 2, (y_min + y_max) / 2, z_avg)
        self.move_camera(
            frame_center=focus_point,
            zoom=1.3,  # Zoom in (higher value = closer view)
            run_time=2.5,
        )
        self.wait(0.5)

        # Fade out bounding box after zoom
        self.play(FadeOut(bbox_lines), FadeOut(bbox_label), run_time=0.5)

        # --- 9) Multi-Step SGD Trajectory (4-5 iterations) ---
        eta = 0.02  # Learning rate (visible with b=10)
        n_steps = 5
        x_traj, y_traj = -0.8, 1.2  # Starting point in focus region

        # Storage for trajectory
        trajectory_points = []
        trajectory_losses = []
        trajectory_dots = []
        trajectory_arrows = []

        # Running annotation panel (fixed in frame)
        step_text = Text("Step: 0/5", font_size=28, color=GOLD).to_corner(UR, buff=0.6).shift(DOWN * 0.5)
        loss_text = Text(f"Loss: {self.rosenbrock(x_traj, y_traj):.2f}", font_size=24, color=GOLD_C).next_to(
            step_text, DOWN, buff=0.2
        )
        annotation_panel = VGroup(step_text, loss_text)
        self.add_fixed_in_frame_mobjects(annotation_panel)
        self.play(FadeIn(annotation_panel), run_time=0.5)

        # Slower camera rotation during SGD trajectory for smoother tracking
        self.begin_ambient_camera_rotation(rate=0.05)

        # Initial point
        z_start = self.rosenbrock(x_traj, y_traj)
        p_start = axes.c2p(x_traj, y_traj, z_start)
        traj_dot = dot3d(p_start, color=GOLD, radius=0.06)
        self.play(FadeIn(traj_dot), run_time=0.4)
        trajectory_points.append((x_traj, y_traj, z_start))
        trajectory_losses.append(z_start)
        trajectory_dots.append(traj_dot)

        # Perform SGD steps
        for step in range(1, n_steps + 1):
            # Compute gradient
            g_step = self.grad_rosenbrock(x_traj, y_traj)

            # Update position
            x_new = x_traj - eta * g_step[0]
            y_new = y_traj - eta * g_step[1]
            z_new = self.rosenbrock(x_new, y_new)
            p_new = axes.c2p(x_new, y_new, z_new)

            # Store trajectory
            trajectory_points.append((x_new, y_new, z_new))
            trajectory_losses.append(z_new)

            # Draw arrow and move dot
            p_current = axes.c2p(x_traj, y_traj, self.rosenbrock(x_traj, y_traj))
            step_arrow = arrow3d(p_current, p_new, color=GOLD_B, stroke_width=6)
            trajectory_arrows.append(step_arrow)

            # Animate movement
            new_dot = dot3d(p_new, color=GOLD, radius=0.06)
            trajectory_dots.append(new_dot)

            # Update annotations
            new_step_text = Text(f"Step: {step}/{n_steps}", font_size=28, color=GOLD)
            new_step_text.move_to(step_text.get_center())
            new_loss_text = Text(f"Loss: {z_new:.2f}", font_size=24, color=GOLD_C)
            new_loss_text.move_to(loss_text.get_center())

            self.play(
                Create(step_arrow),
                FadeIn(new_dot),
                FadeOut(step_text),
                FadeOut(loss_text),
                FadeIn(new_step_text),
                FadeIn(new_loss_text),
                run_time=0.8,
            )

            step_text = new_step_text
            loss_text = new_loss_text
            x_traj, y_traj = x_new, y_new
            self.wait(0.3)

        self.stop_ambient_camera_rotation()

        # --- 10) Draw Complete Trajectory Path with Color Gradient ---
        # Create smooth path from all points with color gradient (YELLOW → GREEN)
        traj_path_lines = VGroup()
        for i in range(len(trajectory_points) - 1):
            p1 = axes.c2p(*trajectory_points[i])
            p2 = axes.c2p(*trajectory_points[i + 1])
            # Color gradient from YELLOW (start) to GREEN (end)
            progress = i / (len(trajectory_points) - 1)
            line_color = interpolate_color(YELLOW, GREEN, progress)
            path_line = Line(p1, p2, color=line_color, stroke_width=5)
            traj_path_lines.add(path_line)

        self.play(Create(traj_path_lines), run_time=1.2)
        self.wait(1.0)

        # Final rotation to appreciate the full trajectory
        self.begin_ambient_camera_rotation(rate=0.06)
        self.wait(4.0)
        self.stop_ambient_camera_rotation()

        # --- 11) Clean Exit ---
        self.play(
            *[FadeOut(dot) for dot in trajectory_dots],
            *[FadeOut(arrow) for arrow in trajectory_arrows],
            FadeOut(traj_path_lines),
            FadeOut(annotation_panel),
            run_time=0.8,
        )
        self.play(
            FadeOut(valley),
            FadeOut(valley_label),
            FadeOut(pt),
            FadeOut(info_panel),
            FadeOut(surface_mobject),
            FadeOut(axes_labels),
            FadeOut(axes),
            run_time=0.8,
        )
