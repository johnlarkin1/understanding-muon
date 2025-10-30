# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository builds visualization tools using Manim to understand the **Muon optimizer** for neural networks. The reference material is in `reference_blog_post.md`, which drives the visualization development.

## Project Structure

```
understanding_muon/
├── viz/              # Manim visualization scenes
│   ├── chapter_01_intro.py    # Full ML training process visualization
│   └── chapter_01_simple.py   # Simple neural network baseline
└── notebooks/        # Jupyter notebooks (likely location)
```

## Development Commands

This project uses **uv** as the package manager. All commands should be prefixed with `uv run`.

### Rendering Manim Visualizations

```bash
# Low quality preview (fast, for development)
uv run manim -pql understanding_muon/viz/chapter_01_intro.py MLTrainingProcess

# High quality render (slow, for final output)
uv run manim -pqh understanding_muon/viz/chapter_01_intro.py MLTrainingProcess

# Other quality flags:
# -pql = preview, quality low
# -pqm = preview, quality medium
# -pqh = preview, quality high
# -p = preview the animation when done
```

### Code Quality

```bash
make format    # Format code with ruff
make lint      # Run ruff and mypy checks
make test      # Render sample scene to verify code works
make check     # Run all checks (format + lint + test)
```

Or run directly:

```bash
uv run ruff format understanding_muon/
uv run ruff check --fix understanding_muon/
uv run mypy understanding_muon/
```

## Manim Visualization Architecture

### Design Philosophy

The visualizations in this project follow these principles:

1. **Abstract over concrete**: Use symbolic mathematical notation (ŷ, y, w_{ij}) rather than specific numerical values
2. **Educational focus**: Each step should clearly illustrate the conceptual process, not implementation details
3. **Progressive revelation**: Show one concept at a time, cleaning up previous content before introducing new formulas
4. **Consistent labeling**: Use "(pred)" and "(target)" descriptors alongside mathematical symbols

### Key Concepts

1. **Scene classes** inherit from `manim.Scene` and implement `construct()` method
2. **VGroup** is used to group related objects (layers, connections, labels)
3. **Animations** are created with `.animate` property or animation classes (`Write`, `FadeIn`, `Create`)
4. **Color hierarchy** uses `interpolate_color()` to create lighter variants (e.g., `BLUE_B`, `BLUE_C`)

### Common Patterns in This Codebase

**Network structure storage:**

- `self.layers` - VGroup containing all neuron layers
- `self.connections` - VGroup containing connection lines between layers
- Each neuron is a VGroup of `(Circle, Text)` - access circle with `neuron_group[0]`

**Animation flow:**

1. Create objects (neurons, connections)
2. Animate their appearance (GrowFromCenter, Create, Write)
3. Transform properties (.animate.set_fill(), .animate.set_color())
4. Clean up (FadeOut when step is complete)

**Step-by-step visualization pattern:**

```python
def create_step_indicator(self):
    """Create visual indicator for current step"""

def highlight_step(self, step_index):
    """Highlight current step, dim others"""

def animate_[step_name](self):
    """Animate a specific training step"""
    # 1. Show step title
    # 2. Perform animations
    # 3. Clean up title
```

### Manim Scene Organization

Each visualization scene should:

- Use color coding to differentiate training steps (GREEN=forward, RED=loss, PURPLE=backward, GOLD=update)
- Store created objects as `self.` attributes for later reference/cleanup
- Use progressive layer-by-layer animations rather than all-at-once
- Include informative labels and annotations
- Clean up temporary elements between steps using FadeOut
- **Use MathTex for LaTeX notation** (e.g., `MathTex(r"\hat{y}")` for ŷ, `MathTex(r"\frac{\partial L}{\partial w}")` for gradients)
- **Position elements to avoid overlap**:
  - Top: Main title
  - Bottom: Step descriptions
  - Right side: Step indicators and notation labels (ŷ, y)
  - Top-left: Mathematical formulas that change between steps
  - Center: Neural network visualization

#### Manim Library Documentation

This project uses Manim for creating animations. Comprehensive Manim documentation
is available in the `manim_docs/` directory:

- `manim_docs/MANIM_API_DOCUMENTATION.md` - Complete API reference (read this for detailed
  information)
- `manim_docs/MANIM_QUICK_REFERENCE.md` - Quick lookup for common tasks
- `manim_docs/DOCUMENTATION_INDEX.md` - Navigation guide

When working with Manim animations, refer to these files for available animations,
mobjects, scenes, colors, and usage patterns.

## Python Environment

- **Python version:** 3.13+
- **Package manager:** uv (not pip/conda)
- **Key dependency:** manim (3Blue1Brown's animation library)

## Linting Configuration

- **Ruff:** Line length 100, Python 3.13 target
- **MyPy:** Permissive settings (no type checking enforcement), ignores missing imports for manim
