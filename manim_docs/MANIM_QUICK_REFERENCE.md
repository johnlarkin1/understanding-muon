# Manim Community - Quick Reference Guide

## Quick Start Example

```python
from manim import *

class BasicScene(Scene):
    def construct(self):
        # Create objects
        circle = Circle(color=BLUE, fill_opacity=0.5)
        text = Text("Hello Manim", font_size=48)
        
        # Animate them
        self.play(Create(circle))
        self.play(Write(text))
        self.wait(1)
        self.play(FadeOut(circle, text))
```

---

## Essential Imports

```python
# Import everything
from manim import *

# Or specific imports
from manim import Scene, Circle, Square, Line
from manim import Create, Transform, FadeIn, FadeOut
from manim import RED, GREEN, BLUE, WHITE
from manim import UP, DOWN, LEFT, RIGHT, ORIGIN
from manim import config, rate_functions
```

---

## Common Animations (Top 20)

| Animation | Purpose |
|-----------|---------|
| `Create(obj)` | Draw an object as if drawing |
| `FadeIn(obj)` | Fade object in from invisible |
| `FadeOut(obj)` | Fade object out to invisible |
| `Transform(obj1, obj2)` | Morph obj1 into obj2 |
| `ReplacementTransform(obj1, obj2)` | Transform and remove original |
| `Write(text)` | Write text as if typing |
| `Indicate(obj)` | Highlight with pulsing effect |
| `Rotate(obj, angle)` | Rotate around center |
| `Move(obj, direction)` | Move in a direction |
| `Scale(obj, factor)` | Scale by factor |
| `GrowFromCenter(obj)` | Grow from center point |
| `FadeToColor(obj, color)` | Fade to specific color |
| `ApplyMethod(obj, method, *args)` | Apply mobject method |
| `ApplyFunction(obj, func)` | Apply transformation function |
| `MoveAlongPath(obj, path)` | Move along a curve |
| `ShowPassingFlash(obj)` | Flash across object |
| `AnimationGroup(*anims)` | Play animations together |
| `Succession(*anims)` | Play animations in sequence |
| `Circumscribe(obj)` | Draw border around object |
| `Blink(obj)` | Quick fade in/out blink |

---

## Common Mobjects (Top 20)

| Mobject | Purpose |
|---------|---------|
| `Circle()` | Circle shape |
| `Square()` | Square shape |
| `Rectangle()` | Rectangle shape |
| `Line(p1, p2)` | Line between points |
| `Arrow(p1, p2)` | Arrow from p1 to p2 |
| `Arc(radius, angle)` | Circular arc |
| `Dot()` | Small point/dot |
| `Text(string)` | Text from font |
| `MathTex(latex)` | LaTeX math expression |
| `Tex(latex)` | General TeX text |
| `Polygon([p1, p2, p3, ...])` | Polygon from points |
| `Triangle()` | Equilateral triangle |
| `Star()` | Star shape |
| `Ellipse()` | Elliptical shape |
| `VGroup(*mobjects)` | Group multiple objects |
| `Axes()` | 2D coordinate axes |
| `NumberPlane()` | Gridded 2D plane |
| `ThreeDAxes()` | 3D coordinate axes |
| `ParametricFunction(func)` | Curve from function |
| `FunctionGraph(func, x_range)` | Graph y=f(x) |

---

## Color Palette (Top 20)

```python
# Primary colors
RED, GREEN, BLUE
PURE_RED, PURE_GREEN, PURE_BLUE

# Accent colors
YELLOW, TEAL, PURPLE, ORANGE, PINK
GOLD, MAROON, BROWN

# Grayscale
WHITE, BLACK, GRAY, LIGHT_GRAY, DARK_GRAY

# Color variants (lighter to darker)
RED_A, RED_B, RED_C, RED_D, RED_E  # (same for other colors)
```

---

## Direction Constants

```python
# Primary directions
UP         # (0, 1, 0)
DOWN       # (0, -1, 0)
RIGHT      # (1, 0, 0)
LEFT       # (-1, 0, 0)
IN         # (0, 0, -1)  - into screen
OUT        # (0, 0, 1)   - out of screen

# Diagonals
UL, UR     # Up-Left, Up-Right
DL, DR     # Down-Left, Down-Right

# Axes
X_AXIS, Y_AXIS, Z_AXIS

# Origin
ORIGIN     # (0, 0, 0)
```

---

## Rate Functions (Easing)

```python
# Basic
rate_func=linear              # Constant speed
rate_func=smooth              # Smooth acceleration
rate_func=smoothstep          # Smooth step

# Ease in/out
rate_func=rate_functions.ease_in_sine
rate_func=rate_functions.ease_out_sine
rate_func=rate_functions.ease_in_out_sine

# Special
rate_func=rate_functions.there_and_back
rate_func=rate_functions.running_start
rate_func=rate_functions.wiggle
rate_func=rate_functions.exponential_decay
```

---

## Configuration

```python
from manim import config

# Display settings
config.frame_height = 8          # Scene height
config.frame_width = 14.22       # Scene width
config.pixel_height = 1080       # Output height
config.pixel_width = 1920        # Output width
config.frame_rate = 60           # FPS

# Quality presets
config.quality = "low_quality"      # 480p @ 15fps
config.quality = "medium_quality"   # 720p @ 30fps
config.quality = "high_quality"     # 1080p @ 60fps
config.quality = "production_quality"  # 1440p @ 60fps

# Output
config.media_dir = "./media"
config.background_color = BLACK
config.write_to_movie = True
config.save_last_frame = False

# Renderer
config.renderer = "cairo"   # or "opengl"
```

---

## Scene Methods

```python
class MyScene(Scene):
    def construct(self):
        # Add objects to scene (no animation)
        self.add(obj)
        
        # Play animations
        self.play(Create(obj), run_time=2)
        
        # Wait without animating
        self.wait(1)
        
        # Remove objects
        self.remove(obj)
        
        # Clear entire scene
        self.clear()
```

---

## Common Mobject Methods

```python
obj = Circle()

# Positioning
obj.move_to(ORIGIN)           # Move to point
obj.shift(RIGHT * 2)          # Shift by vector
obj.align_to(other, UP)       # Align with other

# Sizing
obj.scale(2)                  # Scale by factor
obj.set_width(3)              # Set width
obj.set_height(2)             # Set height

# Appearance
obj.set_color(RED)            # Set stroke color
obj.set_fill(BLUE, opacity=0.5)  # Set fill
obj.set_stroke(width=4, color=RED)  # Set stroke

# Rotation
obj.rotate(PI / 4)            # Rotate by angle
obj.rotate_about_point(angle, point)  # Rotate around point

# Grouping
group = VGroup(obj1, obj2, obj3)
group.arrange(RIGHT)          # Arrange objects
```

---

## Value Tracking & Updates

```python
# Track values for animation
tracker = ValueTracker(0)
number = DecimalNumber(tracker.get_value())

# Update function calls update_func every frame
def update_func(m):
    m.set_value(tracker.get_value())

number.add_updater(update_func)

# Animate the tracker
self.add(number, tracker)
self.play(tracker.animate.set_value(10), run_time=2)
```

---

## Animation Composition

```python
# Play together
self.play(
    Create(circle),
    FadeIn(text),
    Rotate(square, PI),
)

# Play in sequence
self.play(Create(obj1))
self.wait(0.5)
self.play(Transform(obj1, obj2))

# Succession - play one after another
anim1 = Create(obj1)
anim2 = Transform(obj1, obj2)
anim3 = FadeOut(obj2)
self.play(Succession(anim1, anim2, anim3))

# Staggered starts
self.play(LaggedStart(
    Create(c1),
    Create(c2),
    Create(c3),
    lag_ratio=0.5
))
```

---

## 3D Scenes

```python
from manim import *

class My3DScene(ThreeDScene):
    def construct(self):
        # 3D mobjects
        cube = Cube()
        sphere = Sphere()
        axes = ThreeDAxes()
        
        self.add(axes, cube)
        
        # Rotate camera
        self.move_camera(phi=75*DEGREES, theta=45*DEGREES)
        self.begin_ambient_camera_rotation(rate=2*DEGREES)
```

---

## Debugging Tips

```python
# Highlight an object
self.play(Indicate(obj))

# Show object bounds
from manim import SurroundingRectangle
rect = SurroundingRectangle(obj)
self.add(rect)

# Print info
print(obj.get_center())
print(obj.get_width())

# Save intermediate frame
self.add(obj)
self.wait(0.001)  # Single frame
```

---

## File Structure

```
my_project/
├── scenes.py          # Your scene classes
└── media/             # Generated files
    ├── videos/        # Video outputs
    ├── images/        # Image frames
    └── Tex/           # LaTeX temporary files
```

---

## Command Line Usage

```bash
# Render a scene
manim scenes.py MyScene

# Output options
manim -p scenes.py MyScene           # Preview
manim -s scenes.py MyScene           # Save last frame
manim -a scenes.py MyScene           # All scenes

# Quality
manim -ql scenes.py MyScene          # Low quality
manim -qm scenes.py MyScene          # Medium quality
manim -qh scenes.py MyScene          # High quality
manim -qk scenes.py MyScene          # 4K quality

# Other
manim -i scenes.py MyScene           # Interactive mode
manim --renderer=opengl scenes.py MyScene  # Use OpenGL
```

---

## Resources

- Documentation: https://docs.manim.community/
- Examples: https://github.com/ManimCommunity/manim/tree/main/example_scenes
- Community: https://discord.gg/manimcommunity
- Reference: https://docs.manim.community/en/stable/reference.html

