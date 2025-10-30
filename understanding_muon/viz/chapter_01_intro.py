"""
Improved Manim visualization of the 4-step machine learning training process.

Enhanced features:
- Better spacing to prevent label overlap
- Clearer visual hierarchy
- Improved color coding and animations
- More informative annotations
- Smoother transitions between steps

Steps visualized:
1. Forward pass (feeding data in)
2. Loss function (measuring performance)
3. Backward pass (computing gradients)
4. Gradient descent (updating weights)

Usage:
    manim -pql understanding_muon/viz/chapter_01_intro.py MLTrainingProcess
    manim -pqh understanding_muon/viz/chapter_01_intro.py MLTrainingProcess  # for high quality
"""

import numpy as np
from manim import (
    BLUE,
    DARK_GREY,
    DOWN,
    GOLD,
    GRAY,
    GREEN,
    LEFT,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    Scene,
    Text,
    VGroup,
    Write,
    interpolate_color,
)

# Define custom color variants for better visual hierarchy
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
DARK_GRAY = DARK_GREY  # Alias for consistency


class MLTrainingProcess(Scene):
    def construct(self):
        # Title with better positioning
        title = Text("Machine Learning Training Process", font_size=44, weight="BOLD")
        subtitle = Text("4 Key Steps in Neural Network Training", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.2))
        self.wait(1)

        # Move title to top with animation
        self.play(title_group.animate.scale(0.7).to_edge(UP, buff=0.3), run_time=1)
        self.wait(0.5)

        # Create the neural network with better spacing
        self.create_neural_network()
        self.wait(1.5)

        # Create step indicator on the side
        self.create_step_indicator()

        # Execute all 4 training steps
        self.wait(1)

        # Step 1: Forward Pass
        self.highlight_step(0)
        self.animate_forward_pass()
        self.wait(1)

        # Step 2: Loss Function
        self.highlight_step(1)
        self.animate_loss()
        self.wait(1)

        # Step 3: Backward Pass
        self.highlight_step(2)
        self.animate_backward_pass()
        self.wait(1)

        # Step 4: Gradient Descent
        self.highlight_step(3)
        self.animate_gradient_descent()
        self.wait(2)

    def create_step_indicator(self):
        """Create a visual step indicator on the right side."""
        self.step_indicators = VGroup()
        steps = [
            ("1. Forward Pass", GREEN),
            ("2. Loss Function", RED),
            ("3. Backward Pass", PURPLE),
            ("4. Weight Update", GOLD),
        ]

        for i, (step_name, color) in enumerate(steps):
            indicator = Text(step_name, font_size=20, color=color)
            indicator.move_to(RIGHT * 5 + UP * (1.5 - i * 0.8))
            indicator.set_opacity(0.5)
            self.step_indicators.add(indicator)

        self.play(FadeIn(self.step_indicators))

    def highlight_step(self, step_index):
        """Highlight the current step in the indicator."""
        animations = []
        for i, indicator in enumerate(self.step_indicators):
            if i == step_index:
                animations.append(indicator.animate.set_opacity(1).scale(1.1))
            else:
                animations.append(indicator.animate.set_opacity(0.3).scale(1))
        self.play(*animations, run_time=0.5)

    def create_neural_network(self):
        """Create a simple 3-layer neural network with improved spacing."""
        # Network architecture: 3 -> 4 -> 3 -> 2
        layer_sizes = [3, 4, 3, 2]
        self.layers = VGroup()
        self.connections = VGroup()

        # Enhanced spacing parameters
        layer_spacing = 3.0  # Increased from 2.5
        neuron_radius = 0.25  # Slightly larger
        vertical_spacing = 1.0  # Increased vertical spacing

        # Position network more to the left to make room for annotations
        network_offset = LEFT

        # Create layers with better positioning
        for i, size in enumerate(layer_sizes):
            layer = VGroup()
            x_pos = -4.5 + i * layer_spacing  # Start further left

            # Calculate vertical positions with better spacing
            total_height = (size - 1) * vertical_spacing
            start_y = total_height / 2

            for j in range(size):
                y_pos = start_y - j * vertical_spacing

                # Create neuron with gradient fill
                neuron = Circle(
                    radius=neuron_radius, color=BLUE_B, fill_opacity=0.4, stroke_width=2
                )
                neuron.move_to([x_pos, y_pos, 0])
                neuron.shift(network_offset)

                # Add small label inside neuron
                neuron_label = Text(f"{j + 1}", font_size=14, color=WHITE)
                neuron_label.move_to(neuron.get_center())

                layer.add(VGroup(neuron, neuron_label))

            self.layers.add(layer)

        # Create connections with varying opacity based on "weight"
        np.random.seed(42)  # For reproducible "weights"
        for i in range(len(self.layers) - 1):
            layer_connections = VGroup()
            for neuron1_group in self.layers[i]:
                neuron1 = neuron1_group[0]  # Get the circle, not the label
                for neuron2_group in self.layers[i + 1]:
                    neuron2 = neuron2_group[0]

                    # Random opacity to simulate different weight magnitudes
                    opacity = np.random.uniform(0.2, 0.6)

                    line = Line(
                        neuron1.get_center(),
                        neuron2.get_center(),
                        stroke_width=1.5,
                        stroke_opacity=opacity,
                        color=GRAY,
                    )
                    layer_connections.add(line)
            self.connections.add(layer_connections)

        # Animate creation with cascade effect - nodes first, then edges
        for layer in self.layers:
            self.play(
                LaggedStart(
                    *[GrowFromCenter(neuron_group) for neuron_group in layer], lag_ratio=0.1
                ),
                run_time=0.7,
            )

        for connection_layer in self.connections:
            self.play(
                LaggedStart(*[Create(line) for line in connection_layer], lag_ratio=0.01),
                run_time=0.7,
            )

        # Add layer labels with better positioning
        labels = ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"]
        self.layer_labels = VGroup()

        for i, (layer, label) in enumerate(zip(self.layers, labels)):
            text = Text(label, font_size=18, color=BLUE_C)
            # Position below the layer with more spacing
            text.next_to(layer, DOWN, buff=0.6)
            self.layer_labels.add(text)

        self.play(
            LaggedStart(*[Write(label) for label in self.layer_labels], lag_ratio=0.2), run_time=1
        )

        # Add dimension annotations
        self.add_dimension_annotations()

    def add_dimension_annotations(self):
        """Add small dimension annotations to show tensor shapes."""
        dim_annotations = VGroup()

        dimensions = ["(batch, 3)", "(batch, 4)", "(batch, 3)", "(batch, 2)"]
        colors = [YELLOW_B, YELLOW_B, YELLOW_B, YELLOW_B]

        for layer, dim, color in zip(self.layers, dimensions, colors):
            annotation = Text(dim, font_size=12, color=color)
            annotation.next_to(layer, UP, buff=0.3)
            dim_annotations.add(annotation)

        self.play(FadeIn(dim_annotations, shift=DOWN * 0.2))
        self.dim_annotations = dim_annotations

    def animate_forward_pass(self):
        """Animate data flowing forward through the network."""
        # Show title for this step at bottom of frame
        step_title = Text("Forward Pass: Data flows through network", font_size=24, color=GREEN)
        step_title.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(step_title, shift=UP * 0.2))

        # Create data points entering input layer
        data_points = VGroup()
        for neuron_group in self.layers[0]:
            neuron = neuron_group[0]
            data_dot = Dot(color=GREEN_B, radius=0.15)
            data_dot.move_to(neuron.get_center() + LEFT * 1.5)
            data_points.add(data_dot)

        # Animate data entering
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in data_points], lag_ratio=0.2))

        # Move data into input neurons
        animations = []
        for dot, neuron_group in zip(data_points, self.layers[0]):
            neuron = neuron_group[0]
            animations.append(dot.animate.move_to(neuron.get_center()))
        self.play(*animations)

        # Activate each layer progressively
        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]

            # Light up neurons in this layer
            neuron_animations = []
            for neuron_group in layer:
                neuron = neuron_group[0]
                neuron_animations.append(neuron.animate.set_fill(GREEN_B, opacity=0.8))

            self.play(*neuron_animations, run_time=0.5)

            # If not the last layer, show activation flowing to next layer
            if layer_idx < len(self.layers) - 1:
                connection_layer = self.connections[layer_idx]

                # Pulse connections
                self.play(
                    *[
                        line.animate.set_color(GREEN_C).set_stroke(width=3)
                        for line in connection_layer
                    ],
                    run_time=0.4,
                )

                # Reset connections
                self.play(
                    *[
                        line.animate.set_color(GRAY).set_stroke(width=1.5)
                        for line in connection_layer
                    ],
                    run_time=0.2,
                )

        # Show symbolic output notation (y-hat for prediction) on right side under step indicators
        output_label = MathTex(r"\hat{y}", font_size=48, color=GREEN_B)
        output_label.to_edge(RIGHT, buff=2.0).shift(DOWN * 2)  # Increased buffer from wall
        self.play(Write(output_label))
        self.output_label = output_label

        # Add (pred) descriptor
        pred_text = Text("(pred)", font_size=16, color=GREEN_C)
        pred_text.next_to(output_label, RIGHT, buff=0.3)
        self.play(FadeIn(pred_text))
        self.pred_text = pred_text

        # Show input notation on the left
        input_label = MathTex(r"x", font_size=48, color=GREEN_C)
        input_label.next_to(self.layers[0], LEFT, buff=0.8)
        self.play(FadeIn(input_label))
        self.input_label = input_label

        self.wait(0.5)
        self.play(FadeOut(step_title), FadeOut(data_points))

    def animate_loss(self):
        """Visualize the loss function and error computation."""
        step_title = Text("Loss Function: Measure prediction error", font_size=24, color=RED)
        step_title.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(step_title, shift=UP * 0.2))

        # Show target notation (y) under ŷ on right side
        target_label = MathTex(r"y", font_size=48, color=RED_B)
        target_label.next_to(self.output_label, DOWN, buff=0.8)  # Increased buffer between labels
        self.play(Write(target_label))

        # Add small descriptive text
        target_text = Text("(target)", font_size=16, color=RED_C)
        target_text.next_to(target_label, RIGHT, buff=0.3)
        self.play(FadeIn(target_text))

        # Show loss function formula: L(\hat{y}, y)
        loss_formula = MathTex(r"L(\hat{y}, y)", font_size=40, color=RED)
        loss_formula.to_edge(LEFT, buff=0.75).shift(UP * 3)

        # Create box around loss formula
        loss_box = Rectangle(
            height=loss_formula.height + 0.4,
            width=loss_formula.width + 0.6,
            color=RED,
            fill_opacity=0.15,
            stroke_width=3,
        )
        loss_box.move_to(loss_formula.get_center())

        self.play(GrowFromCenter(loss_box))
        self.play(Write(loss_formula))

        # Highlight output neurons with error
        for neuron_group in self.layers[-1]:
            neuron = neuron_group[0]
            self.play(neuron.animate.set_fill(RED_B, opacity=0.8), run_time=0.3)

        # Store all elements that should be cleared at next step
        self.target_label = target_label
        self.target_text = target_text
        self.top_left_content = VGroup(loss_formula, loss_box)

        self.wait(0.5)
        self.play(FadeOut(step_title))

    def animate_backward_pass(self):
        """Animate gradients flowing backward through the network."""
        # Clear previous top-left content (loss formula)
        if hasattr(self, "top_left_content"):
            self.play(FadeOut(self.top_left_content))

        step_title = Text("Backward Pass: Compute gradients", font_size=24, color=PURPLE)
        step_title.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(step_title, shift=UP * 0.2))

        # Show the loss propagating backward from output
        gradient_annotations = VGroup()

        # Propagate gradients backward through layers
        for layer_idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_idx]

            # Highlight neurons receiving gradients
            neuron_animations = []
            for neuron_group in layer:
                neuron = neuron_group[0]
                neuron_animations.append(neuron.animate.set_fill(PURPLE_B, opacity=0.8))

            self.play(*neuron_animations, run_time=0.5)

            # If not the first layer, show gradients flowing backward through connections
            if layer_idx > 0:
                connection_layer = self.connections[layer_idx - 1]

                # Pulse connections in reverse
                self.play(
                    *[
                        line.animate.set_color(PURPLE_C).set_stroke(width=3)
                        for line in connection_layer
                    ],
                    run_time=0.4,
                )

                # Add gradient notation on a few sample connections
                if layer_idx == len(self.layers) - 1:
                    # Show ∂L/∂θ on one connection from last hidden to output
                    sample_line = connection_layer[len(connection_layer) // 2]
                    gradient_label = MathTex(
                        r"\frac{\partial L}{\partial \theta}", font_size=24, color=PURPLE
                    )
                    gradient_label.next_to(sample_line.get_center(), UP, buff=0.3)
                    self.play(FadeIn(gradient_label, scale=0.5))
                    gradient_annotations.add(gradient_label)

                # Reset connections
                self.play(
                    *[
                        line.animate.set_color(GRAY).set_stroke(width=1.5)
                        for line in connection_layer
                    ],
                    run_time=0.2,
                )

        # Show gradient computation summary on the left
        gradient_summary = MathTex(
            r"\frac{\partial L}{\partial \theta}", font_size=36, color=PURPLE_B
        )
        gradient_summary.to_edge(LEFT, buff=0.75).shift(UP * 3)

        gradient_text = Text("computed via\nchain rule", font_size=16, color=PURPLE_C)
        gradient_text.next_to(gradient_summary, DOWN, buff=0.3)

        self.play(Write(gradient_summary))
        self.play(FadeIn(gradient_text))

        # Store top-left content for clearing in next step
        self.top_left_content = VGroup(gradient_annotations, gradient_summary, gradient_text)

        self.wait(0.5)
        self.play(FadeOut(step_title))

    def animate_gradient_descent(self):
        """Animate weight updates via gradient descent."""
        # Clear previous top-left content (gradient formulas)
        if hasattr(self, "top_left_content"):
            self.play(FadeOut(self.top_left_content))

        step_title = Text("Gradient Descent: Update weights", font_size=24, color=GOLD)
        step_title.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(step_title, shift=UP * 0.2))

        # Update equation with proper LaTeX (using theta for all parameters)
        update_eq = MathTex(
            r"\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}",
            font_size=32,
            color=GOLD_B,
        )
        update_eq.to_edge(LEFT, buff=0.75).shift(UP * 3)
        self.play(Write(update_eq))

        # Select specific connections to label with θ_{ij} notation
        param_labels = VGroup()
        labeled_connections = []

        # Label one connection from each layer transition
        connection_indices = [
            (0, len(self.connections[0]) // 3),  # Input to Hidden1
            (1, len(self.connections[1]) // 2),  # Hidden1 to Hidden2
            (2, len(self.connections[2]) // 2),  # Hidden2 to Output
        ]

        for layer_idx, conn_idx in connection_indices:
            if layer_idx < len(self.connections):
                connection_layer = self.connections[layer_idx]
                if conn_idx < len(connection_layer):
                    line = connection_layer[conn_idx]
                    labeled_connections.append((layer_idx, conn_idx, line))

                    # Create parameter label θ_{ij} with layer-specific indices
                    i = layer_idx + 1
                    j = (conn_idx % 3) + 1
                    param_label = MathTex(rf"\theta_{{{i}{j}}}", font_size=20, color=GOLD_B)
                    param_label.next_to(
                        line.get_center(), UP if layer_idx % 2 == 0 else DOWN, buff=0.2
                    )
                    param_labels.add(param_label)

        # Show parameter labels
        self.play(
            LaggedStart(*[FadeIn(label, scale=0.5) for label in param_labels], lag_ratio=0.2)
        )

        # Animate weight updates - change connection appearance
        np.random.seed(43)  # Different seed for "updated" weights
        for connection_layer in self.connections:
            animations = []
            for line in connection_layer:
                # New opacity representing updated weight
                new_opacity = np.random.uniform(0.3, 0.7)
                new_width = np.random.uniform(1.0, 2.5)

                animations.append(
                    line.animate.set_stroke(color=GOLD_C, width=new_width, opacity=new_opacity)
                )

            self.play(*animations, run_time=0.6)

        # Reset to gray but keep new widths/opacities
        for connection_layer in self.connections:
            self.play(*[line.animate.set_color(GRAY) for line in connection_layer], run_time=0.4)

        # Show that loss has improved
        improved_text = Text("Loss decreased", font_size=18, color=GREEN_B)
        improved_text.next_to(update_eq, DOWN, buff=0.5)
        self.play(FadeIn(improved_text))

        # Reset neuron colors to original
        for layer in self.layers:
            for neuron_group in layer:
                neuron = neuron_group[0]
                self.play(neuron.animate.set_fill(BLUE_B, opacity=0.4), run_time=0.3)

        self.wait(0.5)
        # Clean up all remaining elements
        self.play(
            FadeOut(step_title),
            FadeOut(self.output_label),
            FadeOut(self.pred_text),
            FadeOut(self.input_label),
            FadeOut(self.target_label),
            FadeOut(self.target_text),
            FadeOut(update_eq),
            FadeOut(param_labels),
            FadeOut(improved_text),
        )
