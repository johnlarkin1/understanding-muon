"""
Simple neural network visualization - starting point.

Usage:
    manim -pql understanding_muon/viz/chapter_01_simple.py SimpleNetwork
    manim -pqh understanding_muon/viz/chapter_01_simple.py SimpleNetwork
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GRAY,
    LEFT,
    RIGHT,
    UP,
    WHITE,
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

# Color variants
BLUE_B = interpolate_color(BLUE, WHITE, 0.3)
BLUE_C = interpolate_color(BLUE, WHITE, 0.5)


class SimpleNetwork(Scene):
    """Simple, centered neural network visualization."""

    def construct(self):
        # Title
        title = Text("Neural Network", font_size=44, weight="BOLD")
        subtitle = Text("3-Layer Architecture", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP, buff=0.5)

        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle, shift=UP * 0.2))
        self.wait(0.5)

        # Create the neural network
        self.create_neural_network()
        self.wait(2)

    def create_neural_network(self):
        """Create a simple centered neural network."""
        # Network architecture: 3 -> 4 -> 3 -> 2
        layer_sizes = [3, 4, 3, 2]
        self.layers = VGroup()
        self.connections = VGroup()

        # Spacing parameters
        layer_spacing = 2.5
        neuron_radius = 0.25
        vertical_spacing = 1.0

        # Calculate total width to center the network
        total_width = (len(layer_sizes) - 1) * layer_spacing
        start_x = -total_width / 2

        # Create layers
        for i, size in enumerate(layer_sizes):
            layer = VGroup()
            x_pos = start_x + i * layer_spacing

            # Calculate vertical positions
            total_height = (size - 1) * vertical_spacing
            start_y = total_height / 2

            for j in range(size):
                y_pos = start_y - j * vertical_spacing

                # Create neuron
                neuron = Circle(
                    radius=neuron_radius, color=BLUE_B, fill_opacity=0.4, stroke_width=2
                )
                neuron.move_to([x_pos, y_pos, 0])

                # Add label inside neuron
                neuron_label = Text(f"{j + 1}", font_size=14, color=WHITE)
                neuron_label.move_to(neuron.get_center())

                layer.add(VGroup(neuron, neuron_label))

            self.layers.add(layer)

        # Create connections
        np.random.seed(42)
        for i in range(len(self.layers) - 1):
            layer_connections = VGroup()
            for neuron1_group in self.layers[i]:
                neuron1 = neuron1_group[0]
                for neuron2_group in self.layers[i + 1]:
                    neuron2 = neuron2_group[0]

                    # Random opacity for weight variation
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

        # Animate creation - nodes first, then edges
        for layer in self.layers:
            self.play(
                LaggedStart(
                    *[GrowFromCenter(neuron_group) for neuron_group in layer], lag_ratio=0.1
                ),
                run_time=0.5,
            )

        for connection_layer in self.connections:
            self.play(
                LaggedStart(*[Create(line) for line in connection_layer], lag_ratio=0.01),
                run_time=0.5,
            )

        # Add layer labels
        labels = ["Input\nLayer", "Hidden\nLayer 1", "Hidden\nLayer 2", "Output\nLayer"]
        self.layer_labels = VGroup()

        for layer, label in zip(self.layers, labels):
            text = Text(label, font_size=16, color=BLUE_C)
            text.next_to(layer, DOWN, buff=0.8)
            self.layer_labels.add(text)

        self.play(
            LaggedStart(*[Write(label) for label in self.layer_labels], lag_ratio=0.2), run_time=1
        )
