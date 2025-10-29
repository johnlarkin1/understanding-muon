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
    FadeIn,
    GrowFromCenter,
    LaggedStart,
    Line,
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

        # self.wait(1.5)

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
