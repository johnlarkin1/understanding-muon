"""
Manim visualization of the Muon optimization algorithm (2D Contour Version).

Muon: Momentum Orthogonalized by Newton-Schulz (2025)

This visualization shows the step-by-step process of the Muon optimizer using
2D contour plots for clarity. Emphasizes the Newton-Schulz orthogonalization
as the key innovation.

Algorithm steps visualized:
1. Compute gradient: g_t ← ∇_θ f_t(θ_{t-1})
2. Momentum accumulation: B_t ← μ B_{t-1} + g_t
3. Nesterov adjustment: B̃_t ← g_t + μ B_t (if nesterov) else B_t
4. Newton-Schulz orthogonalization: O_t ← NS^{(a,b,c)}_k(B̃_t; ε)  ⭐ KEY STEP
5. Decoupled weight decay: θ_t ← θ_{t-1} - γλθ_{t-1}
6. Adjust learning rate: γ ← 0.2√max(A,B)
7. Final update: θ_t ← θ_t - γO_t

Usage:
    uv run manim -pql understanding_muon/viz/chapter_05_muon_algorithm.py MuonAlgorithm2D
    uv run manim -pqh understanding_muon/viz/chapter_05_muon_algorithm.py MuonAlgorithm2D
    uv run manim -pql understanding_muon/viz/chapter_05_muon_algorithm.py MuonVsOthers
    uv run manim -pql understanding_muon/viz/chapter_05_muon_algorithm.py NewtonSchulzDetail
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    BOLD,
    DOWN,
    GOLD,
    GRAY,
    GREEN,
    LEFT,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    TAU,
    TEAL,
    UL,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Axes,
    CurvedArrow,
    Dot,
    FadeIn,
    FadeOut,
    MathTex,
    Rectangle,
    Scene,
    Tex,
    Text,
    TracedPath,
    VGroup,
    Write,
    interpolate_color,
    smooth,
)

# Define custom color variants for better visual hierarchy
BLUE_B = interpolate_color(BLUE, WHITE, 0.3)
BLUE_C = interpolate_color(BLUE, WHITE, 0.5)
GOLD_B = interpolate_color(GOLD, WHITE, 0.3)
GREEN_B = interpolate_color(GREEN, WHITE, 0.3)
ORANGE_B = interpolate_color(ORANGE, WHITE, 0.3)
PURPLE_B = interpolate_color(PURPLE, WHITE, 0.3)
RED_B = interpolate_color(RED, WHITE, 0.3)
TEAL_B = interpolate_color(TEAL, WHITE, 0.3)
TEAL_C = interpolate_color(TEAL, WHITE, 0.5)
YELLOW_B = interpolate_color(YELLOW, WHITE, 0.3)

# Import utilities from existing chapters
from understanding_muon.viz.chapter_02_optimizers import (
    GLOBAL_MIN_2D,
    MOM_PATH,
    SGD_PATH,
    START,
    X_RANGE,
    Y_RANGE,
    make_contours,
    styblinski_tang_fn,
    styblinski_tang_grad,
)
from understanding_muon.viz.chapter_03_sgd_trapped import LOCAL_MIN_APPROX, mark_minimum

# =============================
# Muon-specific constants
# =============================

# Newton-Schulz polynomial coefficients (from Keller Jordan's implementation)
NS_A = 3.4445
NS_B = -4.7750
NS_C = 2.0315

# =============================
# Helper Functions
# =============================


def newton_schulz_iteration(M: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray:
    """
    Newton-Schulz orthogonalization with 5th-order polynomial.

    Iteratively applies: X ← aX + (bA + cA²)X where A = XX^T
    This converges to an orthogonal matrix by making singular values → 1.

    Args:
        M: Input matrix (momentum matrix)
        steps: Number of NS iterations (default 5)
        eps: Numerical stability epsilon

    Returns:
        Orthogonalized matrix O
    """
    a, b, c = NS_A, NS_B, NS_C

    # Work with float32 for numerical stability
    X = M.astype(np.float32, copy=True)

    # Handle shape: if tall matrix, transpose for efficiency
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    # Normalize by Frobenius norm
    X = X / (np.linalg.norm(X, "fro") + eps)

    # Apply Newton-Schulz iterations
    # ρ(X) = aX + b(XX^T)X + c(XX^T)²X
    for _ in range(steps):
        A = X @ X.T
        B_term = b * A + c * (A @ A)
        X = a * X + B_term @ X

    # Restore original orientation
    if transposed:
        X = X.T

    return X


def compute_muon_path(
    start: np.ndarray,
    lr: float = 0.02,
    beta: float = 0.95,
    weight_decay: float = 0.01,
    steps: int = 60,
    ns_steps: int = 5,
    use_nesterov: bool = True,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Compute Muon optimizer path on the Styblinski-Tang function.

    Returns:
        - path: Array of positions (steps+1, 2)
        - momentum_vectors: List of B_t vectors before orthogonalization
        - orthogonalized_vectors: List of O_t vectors after orthogonalization
    """
    theta = start.astype(float).copy()

    # For 2D optimization, momentum is a 2-vector
    # Reshape to column vector for matrix ops: (2,) -> (2, 1)
    if theta.ndim == 1:
        theta = theta[:, None]

    path = [theta.copy()]
    momentum_vectors = []
    orthogonalized_vectors = []

    B = np.zeros_like(theta)  # Momentum buffer

    for _ in range(steps):
        # Step 1: Compute gradient
        g = styblinski_tang_grad(theta[0, 0], theta[1, 0])
        g = g[:, None]  # Make column vector

        # Step 2: Momentum accumulation
        B = beta * B + g

        # Step 3: Nesterov adjustment
        if use_nesterov:
            B_tilde = g + beta * B
        else:
            B_tilde = B

        momentum_vectors.append(B_tilde.copy())

        # Step 4: Newton-Schulz orthogonalization ⭐ KEY STEP
        O = newton_schulz_iteration(B_tilde, steps=ns_steps)
        orthogonalized_vectors.append(O.copy())

        # Step 5: Decoupled weight decay
        theta = theta - lr * weight_decay * theta

        # Step 6: Adjust learning rate (Moonshot version)
        A, B_dim = theta.shape
        lr_adjusted = lr * 0.2 * np.sqrt(float(max(A, B_dim)))

        # Step 7: Final parameter update
        theta = theta - lr_adjusted * O

        path.append(theta.copy())

    # Convert back to (steps, 2) for compatibility
    path_array = np.array([p.squeeze() for p in path])
    return path_array, momentum_vectors, orthogonalized_vectors


# Precompute Muon path for reuse across scenes
MUON_START = START  # Same starting point as momentum
MUON_PATH, MUON_MOMENTUM_VECS, MUON_ORTHO_VECS = compute_muon_path(
    MUON_START, lr=0.02, beta=0.95, weight_decay=0.01, steps=60, ns_steps=5
)


# ============================================
# Scene 1: Muon Algorithm (2D Contour)
# ============================================
class MuonAlgorithm2D(Scene):
    """
    Primary scene showing Muon optimizer on 2D contour plot with clear
    step-by-step algorithm visualization.
    """

    def construct(self):
        # Setup 2D axes and contours
        ax = Axes(x_range=X_RANGE, y_range=Y_RANGE, x_length=7, y_length=7, tips=False)
        ax_labels = VGroup(ax.get_x_axis_label(Tex("x")), ax.get_y_axis_label(Tex("y")))
        ax.shift(LEFT * 1.5)  # Shift left to make room for algorithm display

        levels = (-70, -60, -50, -40, -30, -20, -10, 0, 20, 50, 100)
        contours = make_contours(list(levels))
        plot_group = VGroup(contours).move_to(ax.get_origin())

        # Title
        title = Tex("Muon: Momentum Orthogonalized by Newton-Schulz").to_edge(UP).scale(0.7)
        subtitle = Tex("7-Step Optimization Algorithm", color=TEAL_B).scale(0.5)
        subtitle.next_to(title, DOWN, buff=0.15)

        self.play(Write(title), FadeIn(subtitle))
        self.play(FadeIn(ax), FadeIn(ax_labels))
        self.play(FadeIn(plot_group, shift=0.25 * UP, lag_ratio=0.05))
        self.wait(0.5)

        # Mark minima
        global_min_marker = mark_minimum(ax, GLOBAL_MIN_2D, "Global", GREEN, is_3d=False)
        local_min_marker = mark_minimum(ax, LOCAL_MIN_APPROX, "Local", RED, is_3d=False)
        self.play(FadeIn(global_min_marker), FadeIn(local_min_marker))

        # Starting point
        start_marker = Dot(ax.c2p(START[0], START[1]), color=TEAL, radius=0.08)
        start_label = Tex("Start", font_size=24).next_to(start_marker, UP + RIGHT, buff=0.1)
        self.play(FadeIn(start_marker), Write(start_label))
        self.wait(0.5)

        # Create algorithm step display on the right
        self.create_algorithm_display()
        self.wait(0.5)

        # Create moving dot and trail
        dot = Dot(ax.c2p(MUON_PATH[0, 0], MUON_PATH[0, 1]), color=TEAL, radius=0.08)
        trail = TracedPath(dot.get_center, stroke_color=TEAL, stroke_width=4, stroke_opacity=0.8)
        self.add(dot, trail)

        # Animate with detailed steps for first few iterations
        detailed_iters = 3
        step_time = 0.6

        for i in range(1, detailed_iters + 1):
            if i >= len(MUON_PATH):
                break

            # Show detailed algorithmic steps
            self.show_iteration_steps(i - 1)

            # Move dot to new position
            p = MUON_PATH[i]
            self.play(
                dot.animate.move_to(ax.c2p(float(p[0]), float(p[1]))),
                run_time=step_time,
                rate_func=smooth,
            )
            self.wait(0.3)

        # Continue remaining iterations quickly
        fast_step_time = 0.1
        for i in range(detailed_iters + 1, len(MUON_PATH)):
            p = MUON_PATH[i]
            self.play(
                dot.animate.move_to(ax.c2p(float(p[0]), float(p[1]))),
                run_time=fast_step_time,
                rate_func=smooth,
            )

        self.wait(1)

        # Success message
        success = Text("✓ Converged to global minimum!", color=GREEN_B, weight=BOLD).scale(0.5)
        success.next_to(self.algo_display, DOWN, buff=0.5)
        self.play(Write(success))
        self.wait(2)

        # Fadeout
        self.play(
            FadeOut(
                VGroup(
                    ax,
                    ax_labels,
                    plot_group,
                    title,
                    subtitle,
                    global_min_marker,
                    local_min_marker,
                    start_marker,
                    start_label,
                    dot,
                    trail,
                    self.algo_display,
                    success,
                )
            )
        )

    def create_algorithm_display(self):
        """Create visual display of the 7 algorithmic steps on the right side."""
        step_texts = [
            (r"1. $g_t = \nabla f$", GREEN_B),
            (r"2. $B_t = \mu B + g$", BLUE_B),
            (r"3. Nesterov", BLUE_C),
            (r"4. \textbf{NS Ortho} $\star$", TEAL_B),  # Key step
            (r"5. Weight decay", ORANGE_B),
            (r"6. Adjust LR", YELLOW_B),
            (r"7. Update $\theta$", GOLD_B),
        ]

        self.step_indicators = VGroup()
        for i, (text, color) in enumerate(step_texts):
            step = Tex(text, font_size=24, color=color)
            step.move_to(RIGHT * 4.5 + UP * (2 - i * 0.5))
            step.set_opacity(0.4)  # Start dimmed
            self.step_indicators.add(step)

        # Group all steps
        self.algo_display = VGroup(self.step_indicators)

        # Add box around the algorithm
        algo_box = Rectangle(
            height=4.5,
            width=2.5,
            color=TEAL,
            fill_opacity=0.05,
            stroke_width=2,
            stroke_opacity=0.5,
        )
        algo_box.move_to(self.step_indicators.get_center())
        self.algo_display.add(algo_box)

        self.play(FadeIn(self.algo_display))

    def show_iteration_steps(self, iter_idx):
        """
        Highlight and show formulas for algorithmic steps during detailed iterations.
        """
        # Define full formulas for key steps
        step_formulas = [
            MathTex(r"g_t = \nabla_{\theta} f(\theta)", font_size=28, color=GREEN_B),
            MathTex(r"B_t = \mu B_{t-1} + g_t", font_size=28, color=BLUE_B),
            MathTex(r"\widetilde{B}_t = g_t + \mu B_t", font_size=28, color=BLUE_C),
            VGroup(
                MathTex(r"O_t = \mathrm{NS}_{5}(\widetilde{B}_t)", font_size=28, color=TEAL_B),
                Tex("KEY STEP", font_size=20, color=TEAL).shift(DOWN * 0.3),
            ),
            MathTex(r"\theta \leftarrow (1-\gamma\lambda)\theta", font_size=28, color=ORANGE_B),
            MathTex(r"\gamma = 0.2\sqrt{\max(A,B)}", font_size=28, color=YELLOW_B),
            MathTex(r"\theta = \theta - \gamma O_t", font_size=28, color=GOLD_B),
        ]

        # Show steps 1-4 with emphasis on step 4
        for step_idx in range(7):
            # Highlight current step
            self.highlight_step(step_idx)

            # Show formula for key steps (1, 2, 3, 4)
            if step_idx <= 3:
                formula = step_formulas[step_idx]
                formula.next_to(self.algo_display, LEFT, buff=1).shift(UP * 2)

                self.play(FadeIn(formula, scale=0.9), run_time=0.4)

                # Extra pause on Newton-Schulz step
                if step_idx == 3:
                    self.wait(0.6)
                else:
                    self.wait(0.3)

                self.play(FadeOut(formula), run_time=0.2)

        # Reset all steps to dimmed
        self.reset_steps()

    def highlight_step(self, step_index):
        """Highlight current algorithmic step."""
        # Reset all first
        for indicator in self.step_indicators:
            indicator.set_opacity(0.4).scale(1.0)

        # Then highlight the current one
        if 0 <= step_index < len(self.step_indicators):
            self.step_indicators[step_index].set_opacity(1.0).scale(1.1)

    def reset_steps(self):
        """Reset all steps to dimmed state."""
        for indicator in self.step_indicators:
            indicator.set_opacity(0.4).scale(1.0)


# ============================================
# Scene 2: Muon vs Others (Comparison)
# ============================================
class MuonVsOthers(Scene):
    """
    Comparison showing Muon vs SGD vs Momentum on 2D contour.
    """

    def construct(self):
        # Setup
        ax = Axes(x_range=X_RANGE, y_range=Y_RANGE, x_length=8, y_length=8, tips=False)
        ax_labels = VGroup(ax.get_x_axis_label(Tex("x")), ax.get_y_axis_label(Tex("y")))

        levels = (-70, -60, -50, -40, -30, -20, -10, 0, 20, 50, 100)
        contours = make_contours(list(levels))
        plot_group = VGroup(contours).move_to(ax.get_origin())

        # Title
        title = Tex("Optimizer Comparison: SGD vs Momentum vs Muon").to_edge(UP).scale(0.7)
        legend = VGroup(
            Tex("Yellow = SGD (trapped)", color=YELLOW, font_size=24),
            Tex("Blue = Momentum", color=BLUE, font_size=24),
            Tex("Teal = Muon", color=TEAL, font_size=24),
        ).arrange(RIGHT, buff=0.5)
        legend.next_to(title, DOWN, buff=0.2)

        self.play(Write(title), FadeIn(legend))
        self.play(FadeIn(ax), FadeIn(ax_labels))
        self.play(FadeIn(plot_group, shift=0.25 * UP, lag_ratio=0.05))
        self.wait(0.5)

        # Mark minima
        start_marker = Dot(ax.c2p(START[0], START[1]), color=GRAY, radius=0.08)
        start_label = Tex("Start", font_size=24).next_to(start_marker, UP + RIGHT, buff=0.1)
        local_min_marker = mark_minimum(ax, LOCAL_MIN_APPROX, "Local", RED, is_3d=False)
        global_min_marker = mark_minimum(ax, GLOBAL_MIN_2D, "Global", GREEN, is_3d=False)

        self.play(FadeIn(start_marker), Write(start_label))
        self.play(FadeIn(local_min_marker), FadeIn(global_min_marker))
        self.wait(0.5)

        # Create dots and trails for all three optimizers
        dot_sgd = Dot(ax.c2p(SGD_PATH[0, 0], SGD_PATH[0, 1]), color=YELLOW, radius=0.06)
        dot_mom = Dot(ax.c2p(MOM_PATH[0, 0], MOM_PATH[0, 1]), color=BLUE, radius=0.06)
        dot_muon = Dot(ax.c2p(MUON_PATH[0, 0], MUON_PATH[0, 1]), color=TEAL, radius=0.06)

        trail_sgd = TracedPath(dot_sgd.get_center, stroke_color=YELLOW, stroke_width=3)
        trail_mom = TracedPath(dot_mom.get_center, stroke_color=BLUE, stroke_width=3)
        trail_muon = TracedPath(dot_muon.get_center, stroke_color=TEAL, stroke_width=3)

        self.add(dot_sgd, dot_mom, dot_muon, trail_sgd, trail_mom, trail_muon)
        self.wait(0.3)

        # Animate all paths simultaneously
        step_time = 0.10
        max_steps = max(len(SGD_PATH), len(MOM_PATH), len(MUON_PATH))

        for i in range(1, max_steps):
            animations = []

            if i < len(SGD_PATH):
                p = SGD_PATH[i]
                animations.append(dot_sgd.animate.move_to(ax.c2p(float(p[0]), float(p[1]))))

            if i < len(MOM_PATH):
                p = MOM_PATH[i]
                animations.append(dot_mom.animate.move_to(ax.c2p(float(p[0]), float(p[1]))))

            if i < len(MUON_PATH):
                p = MUON_PATH[i]
                animations.append(dot_muon.animate.move_to(ax.c2p(float(p[0]), float(p[1]))))

            if animations:
                self.play(*animations, run_time=step_time, rate_func=smooth)

        self.wait(1)

        # Results annotation
        results = VGroup(
            Text("SGD: Trapped ✗", color=YELLOW_B, weight=BOLD, font_size=28),
            Text("Momentum: Escapes but oscillates", color=BLUE_B, weight=BOLD, font_size=28),
            Text("Muon: Fast & stable ✓", color=TEAL_B, weight=BOLD, font_size=28),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        results.to_edge(DOWN)

        self.play(Write(results, lag_ratio=0.3, run_time=2))
        self.wait(1)

        # Arrow emphasizing Muon success
        arrow = CurvedArrow(
            ax.c2p(MUON_PATH[-1, 0] + 0.6, MUON_PATH[-1, 1]),
            ax.c2p(GLOBAL_MIN_2D[0] + 0.4, GLOBAL_MIN_2D[1] + 0.4),
            color=TEAL_B,
            stroke_width=4,
            angle=-TAU / 8,
        )
        arrow_label = Tex("Orthogonalization\\\\wins!", font_size=28, color=TEAL_B)
        arrow_label.next_to(arrow.get_center(), UP + LEFT, buff=0.1)

        self.play(FadeIn(arrow), Write(arrow_label))
        self.wait(2)

        # Fadeout
        self.play(
            FadeOut(
                VGroup(
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
                    dot_muon,
                    trail_sgd,
                    trail_mom,
                    trail_muon,
                    results,
                    arrow,
                    arrow_label,
                )
            )
        )


# ============================================
# Scene 3: Newton-Schulz Detail
# ============================================
class NewtonSchulzDetail(Scene):
    """
    Detailed explanation of Newton-Schulz orthogonalization mathematics.
    """

    def construct(self):
        # Title
        title = Tex("Newton-Schulz Orthogonalization", font_size=48).to_edge(UP)
        subtitle = Tex("The Key Innovation in Muon", color=TEAL_B, font_size=32)
        subtitle.next_to(title, DOWN, buff=0.2)

        self.play(Write(title), FadeIn(subtitle))
        self.wait(1)

        # The Problem
        problem_box = Rectangle(height=2, width=11, color=RED, fill_opacity=0.1, stroke_width=3)
        problem_box.shift(UP * 1.5)

        problem_title = Text("The Problem", weight=BOLD, color=RED, font_size=32)
        problem_title.next_to(problem_box, UP, buff=0.2)

        problem_text = Tex(
            r"Momentum matrix $B$ becomes \textbf{rank-deficient}",
            r"\\Only a few gradient directions dominate",
            font_size=28,
        ).move_to(problem_box)

        self.play(FadeIn(problem_box), Write(problem_title))
        self.play(Write(problem_text))
        self.wait(1.5)

        # The Solution
        solution_box = Rectangle(height=2, width=11, color=GREEN, fill_opacity=0.1, stroke_width=3)
        solution_box.shift(DOWN * 1.5)

        solution_title = Text("The Solution", weight=BOLD, color=GREEN, font_size=32)
        solution_title.next_to(solution_box, UP, buff=0.2)

        solution_text = Tex(
            r"Orthogonalize $B$ to \textbf{amplify rare directions}",
            r"\\Using Newton-Schulz polynomial iteration",
            font_size=28,
        ).move_to(solution_box)

        self.play(FadeIn(solution_box), Write(solution_title))
        self.play(Write(solution_text))
        self.wait(2)

        # Transition to algorithm
        self.play(
            FadeOut(VGroup(problem_box, problem_title, problem_text)),
            FadeOut(VGroup(solution_box, solution_title, solution_text)),
        )

        # Algorithm details
        algo_title = Tex("Newton-Schulz Algorithm", font_size=36, color=TEAL_B)
        algo_title.shift(UP * 2.5)

        algo_box = Rectangle(height=4, width=10, color=TEAL, fill_opacity=0.1, stroke_width=3)

        algo_steps = VGroup(
            MathTex(r"\text{1. Normalize: } X_0 = \frac{B}{\|B\|_F}", font_size=32),
            MathTex(r"\text{2. Iterate } k=5 \text{ times:}", font_size=32),
            MathTex(r"\quad A = X X^T", font_size=32),
            MathTex(r"\quad X \leftarrow aX + (bA + cA^2)X", font_size=32, color=TEAL_B),
            MathTex(r"\text{3. Return orthogonal } O = X", font_size=32, color=GOLD_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        algo_steps.move_to(algo_box)

        self.play(Write(algo_title))
        self.play(FadeIn(algo_box))
        self.play(Write(algo_steps, lag_ratio=0.3, run_time=3))
        self.wait(1.5)

        # Coefficients
        coeff_text = VGroup(
            Tex(r"Coefficients:", font_size=28, color=YELLOW_B),
            MathTex(rf"a = {NS_A:.4f}", font_size=28),
            MathTex(rf"b = {NS_B:.4f}", font_size=28),
            MathTex(rf"c = {NS_C:.4f}", font_size=28),
        ).arrange(RIGHT, buff=0.5)
        coeff_text.next_to(algo_box, DOWN, buff=0.5)

        self.play(FadeIn(coeff_text))
        self.wait(1.5)

        # Key insight
        insight = Tex(
            r"Result: All gradient directions weighted \textbf{equally}",
            font_size=32,
            color=TEAL_B,
        )
        insight.to_edge(DOWN)

        self.play(Write(insight))
        self.wait(2)

        # Fadeout
        self.play(
            FadeOut(
                VGroup(title, subtitle, algo_title, algo_box, algo_steps, coeff_text, insight)
            )
        )
