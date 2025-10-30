# Manim Community Library - Comprehensive API Documentation

## Overview

Manim is a Python library for creating mathematical animations programmatically. This document provides a complete overview of the public API available to external users, organized by category.

---

## 1. Animation Classes

The animation module provides numerous animation classes for bringing mobjects to life. All animation classes inherit from the base `Animation` class.

### Base Animation Class
- **Animation**: The base class for all animations. All other animations inherit from this.
- **Add**: Add a mobject to the scene without animation.
- **Wait**: Wait for a specified duration without animating anything.

### Creation Animations
Creation animations are used to draw or create objects on screen.

- **Create**: Draw an object as if someone is drawing it.
- **Uncreate**: Remove an object by undrawing it (reverse of Create).
- **ShowPartial**: Show part of a mobject based on submobject ranges.
- **DrawBorderThenFill**: Draw the border first, then fill the interior.
- **Write**: Draw text as if someone is writing it (extends DrawBorderThenFill).
- **Unwrite**: Reverse of Write animation.
- **SpiralIn**: Spiral objects in from a center point.
- **ShowIncreasingSubsets**: Show submobjects one by one in increasing fashion.
- **AddTextLetterByLetter**: Add text letter by letter (extends ShowIncreasingSubsets).
- **RemoveTextLetterByLetter**: Remove text letter by letter.
- **ShowSubmobjectsOneByOne**: Show each submobject one at a time.
- **AddTextWordByWord**: Add text word by word using Succession.
- **TypeWithCursor**: Simulate typing with a cursor.
- **UntypeWithCursor**: Reverse of TypeWithCursor.

### Fading Animations
Fade animations transition objects between visible and invisible states.

- **FadeIn**: Gradually make an object visible.
- **FadeOut**: Gradually make an object invisible.

### Transform Animations
Transform animations change one mobject into another or apply transformations.

- **Transform**: Smoothly transform one mobject into another.
- **ReplacementTransform**: Like Transform, but removes the original mobject.
- **TransformFromCopy**: Transform from a copy of the mobject, leaving original intact.
- **ClockwiseTransform**: Transform in a clockwise manner.
- **CounterclockwiseTransform**: Transform in a counterclockwise manner.
- **MoveToTarget**: Move a mobject to a previously set target position.
- **ApplyMethod**: Apply a method to a mobject during animation.
- **ApplyPointwiseFunction**: Apply a function to each point of a mobject.
- **ApplyPointwiseFunctionToCenter**: Apply a function only to the center point.
- **FadeToColor**: Fade a mobject to a specific color.
- **ScaleInPlace**: Scale a mobject while keeping its center fixed.
- **ShrinkToCenter**: Shrink a mobject toward its center.
- **Restore**: Restore a mobject to its previous state.
- **ApplyFunction**: Apply a complex function to transform a mobject.
- **ApplyMatrix**: Apply a matrix transformation.
- **ApplyComplexFunction**: Apply a complex function using complex numbers.
- **CyclicReplace**: Cyclically replace mobjects.
- **Swap**: Swap the positions of two mobjects.
- **FadeTransform**: Fade between two mobjects while transforming.
- **FadeTransformPieces**: Like FadeTransform but for individual pieces.

### Indication Animations
Indication animations are used to highlight or draw attention to objects.

- **FocusOn**: Focus the camera on a specific mobject or point.
- **Indicate**: Highlight a mobject with a subtle pulse or scaling animation.
- **Flash**: Create a brief flash effect.
- **ShowPassingFlash**: Show a moving flash across an object.
- **ShowPassingFlashWithThinningStrokeWidth**: Flash with changing stroke width.
- **ApplyWave**: Apply a wave transformation to an object.
- **Wiggle**: Make an object wiggle back and forth.
- **Circumscribe**: Draw a border around an object.
- **Blink**: Make an object blink (fade in and out quickly).

### Growth Animations
Growth animations expand or contract objects.

- **GrowFromPoint**: Grow an object from a specific point.
- **GrowFromCenter**: Grow an object from its center.
- **GrowFromEdge**: Grow an object from a specific edge.
- **GrowArrow**: Specially designed growth for arrows.
- **SpinInFromNothing**: Spin an object in from nothing.

### Movement Animations
Movement animations change the position of objects.

- **Homotopy**: Apply a continuous deformation (homotopy) to an object.
- **SmoothedVectorizedHomotopy**: Smoothed version of homotopy.
- **ComplexHomotopy**: Homotopy using complex functions.
- **PhaseFlow**: Move objects according to a phase flow field.
- **MoveAlongPath**: Move an object along a specified path.

### Rotation Animations
Rotation animations rotate objects.

- **Rotating**: Continuously rotate an object.
- **Rotate**: Rotate an object by a specified angle.

### Numerical Animations
Animations for numerical values.

- **ChangingDecimal**: Animate decimal number changes.
- **ChangeDecimalToValue**: Change a decimal to a target value.

### Composition Animations
Composition animations combine other animations.

- **AnimationGroup**: Play multiple animations together.
- **Succession**: Play animations one after another in sequence.
- **LaggedStart**: Play animations with staggered start times.
- **LaggedStartMap**: Map animations with lagged starts.

### Specialized Animations
- **Broadcast**: Broadcast an animation to multiple objects.

### Speed Modification
- **ChangeSpeed**: Modify the speed of an animation.

### Transform Matching
- **TransformMatchingAbstractBase**: Base class for matching transformations.
- **TransformMatchingShapes**: Transform based on matching shapes.
- **TransformMatchingTex**: Transform based on matching TeX structures.

### Updaters
- **Update**: Update a mobject based on a function during animation.

---

## 2. Mobject Classes

Mobjects are the mathematical objects that can be animated. They form a hierarchy with `Mobject` as the base class.

### Base Classes
- **Mobject**: The base class for all mathematical objects.
- **Group**: A group of mobjects (legacy, similar to VGroup).
- **VMobject**: Vector-based mobject for smooth curves and shapes.
- **VGroup**: A group container for multiple mobjects.
- **VDict**: A dictionary-like container for mobjects.
- **VectorizedPoint**: A single point in space.

### Geometry Classes

#### Lines and Arrows
- **Line**: A simple line between two points.
- **DashedLine**: A line made up of dashes.
- **TangentLine**: A line tangent to a curve at a point.
- **Elbow**: An elbow/corner shape.
- **Arrow**: A line with an arrowhead.
- **Vector**: A vector arrow from origin.
- **DoubleArrow**: An arrow with heads at both ends.

#### Circles and Arcs
- **Arc**: A circular arc.
- **ArcBetweenPoints**: An arc connecting two points.
- **Circle**: A complete circle.
- **Dot**: A small dot (small circle).
- **AnnotationDot**: A dot with annotation capabilities.
- **LabeledDot**: A dot with a label.
- **Ellipse**: An elliptical shape.
- **CubicBezier**: A cubic Bezier curve.

#### Angles
- **Angle**: An angle shape.
- **RightAngle**: A right angle shape.

#### Sectors and Annuli
- **AnnularSector**: A sector of an annulus (ring).
- **Sector**: A sector of a circle.
- **Annulus**: A ring shape (annulus).

#### Polygons
- **Polygon**: A polygon shape.
- **Polygram**: A generalized polygon.
- **RegularPolygon**: A regular polygon with equal sides.
- **RegularPolygram**: A regular polygram.
- **Triangle**: An equilateral triangle.
- **Rectangle**: A rectangle.
- **Square**: A square.
- **RoundedRectangle**: A rectangle with rounded corners.
- **Star**: A star shape.
- **Cutout**: Create cutouts in shapes.
- **ConvexHull**: Convex hull of points.

#### Arc Polygons
- **ArcPolygon**: A polygon with curved sides.
- **ArcPolygonFromArcs**: Create arc polygon from individual arcs.

#### Curved Elements
- **CurvedArrow**: An arrow with a curved path.
- **CurvedDoubleArrow**: A double-headed curved arrow.

#### Labeled Geometry
- **Label**: A label for geometric objects.
- **LabeledLine**: A line with a label.
- **LabeledArrow**: An arrow with a label.
- **LabeledPolygram**: A polygon with labels.

### Text Mobjects

#### Basic Text
- **Text**: General text mobject using fonts.
- **MarkupText**: Text with markup for styling.
- **Paragraph**: Multiple lines of text.

#### Mathematical Text
- **MathTex**: Mathematical text using LaTeX.
- **Tex**: General TeX text.
- **BulletedList**: A bulleted list of text items.
- **Title**: A title text (extends Tex).
- **SingleStringMathTex**: A single math expression.

#### Numerical Text
- **DecimalNumber**: A decimal number that can be animated.
- **Integer**: An integer number.
- **Variable**: A variable with updatable value.

#### Code
- **Code**: Source code with syntax highlighting.

### Graphing Mobjects

#### Coordinate Systems
- **Axes**: Standard 2D coordinate axes.
- **ThreeDAxes**: 3D coordinate axes.
- **NumberPlane**: A plane with gridlines (extends Axes).
- **PolarPlane**: A plane with polar coordinates.
- **ComplexPlane**: A plane for complex numbers.

#### Functions
- **ParametricFunction**: A parametrically defined curve.
- **FunctionGraph**: A graph of a mathematical function.
- **ImplicitFunction**: A graph of an implicitly defined function.

#### Special Graphing
- **NumberLine**: A 1D number line.
- **UnitInterval**: A number line from 0 to 1.
- **SampleSpace**: A rectangle for probability visualization.
- **BarChart**: A bar chart for data visualization.

#### Graph Theory
- **GenericGraph**: Base class for network-style graphs.
- **Graph**: A directed graph with vertices and edges.
- **DiGraph**: A directed graph.

### 3D Mobjects

#### 3D Geometry
- **ThreeDVMobject**: Base class for 3D vector mobjects.
- **Sphere**: A 3D sphere.
- **Dot3D**: A 3D dot.
- **Cube**: A 3D cube.
- **Prism**: A 3D prism.
- **Cone**: A 3D cone.
- **Cylinder**: A 3D cylinder.
- **Line3D**: A 3D line.
- **Arrow3D**: A 3D arrow.
- **Torus**: A 3D torus (doughnut shape).

#### 3D Surfaces
- **Surface**: A 3D surface.

#### Polyhedra
- **Polyhedron**: A 3D polyhedron.
- **Tetrahedron**: A regular tetrahedron.
- **Octahedron**: A regular octahedron.
- **Icosahedron**: A regular icosahedron.
- **Dodecahedron**: A regular dodecahedron.
- **ConvexHull3D**: Convex hull in 3D.

### Table and Matrix Mobjects

#### Tables
- **Table**: A data table.
- **MathTable**: A table with mathematical expressions.
- **MobjectTable**: A table containing mobjects.
- **IntegerTable**: A table of integers.
- **DecimalTable**: A table of decimal numbers.

#### Matrices
- **Matrix**: A mathematical matrix.
- **DecimalMatrix**: A matrix with decimal values.
- **IntegerMatrix**: A matrix with integer values.
- **MobjectMatrix**: A matrix containing mobjects.

### Vector Fields
- **VectorField**: A vector field visualization.
- **ArrowVectorField**: Vector field using arrows.
- **StreamLines**: Vector field using stream lines.

### Other Mobjects

#### Value Tracking
- **ValueTracker**: Tracks a numerical value for animations.
- **ComplexValueTracker**: Tracks a complex number.

#### SVG and Images
- **SVGMobject**: Create mobjects from SVG files.
- **VMobjectFromSVGPath**: Create mobject from SVG path.
- **ImageMobject**: Display raster images.
- **ImageMobjectFromCamera**: Create image from camera output.

#### Braces and Annotations
- **Brace**: A brace symbol for annotation.
- **BraceBetweenPoints**: A brace between two points.
- **ArcBrace**: A brace following an arc.
- **BraceLabel**: A label attached to a brace.
- **BraceText**: Text attached to a brace.

#### Boolean Operations
- **Union**: Union of two shapes.
- **Difference**: Difference of two shapes.
- **Intersection**: Intersection of two shapes.
- **Exclusion**: Exclusive or of two shapes.

#### Tips/Arrows
- **ArrowTip**: Base class for arrow tips.
- **StealthTip**: A sleek arrow tip.
- **ArrowTriangleTip**: Triangular arrow tip.
- **ArrowTriangleFilledTip**: Filled triangular tip.
- **ArrowCircleTip**: Circular arrow tip.
- **ArrowCircleFilledTip**: Filled circular tip.
- **ArrowSquareTip**: Square arrow tip.
- **ArrowSquareFilledTip**: Filled square tip.

#### Logo
- **Logo**: Manim community logo.

---

## 3. Scene Classes

Scene classes provide the canvas and context for animations.

### Base Scene
- **Scene**: The fundamental scene class where all animations happen.

### Specialized Scenes

#### 3D Scenes
- **ThreeDScene**: Scene for 3D animations with 3D camera.
- **SpecialThreeDScene**: Extended ThreeDScene with additional features.

#### Vector/Linear Algebra
- **VectorScene**: Scene optimized for vector operations.
- **LinearTransformationScene**: Scene for demonstrating linear transformations.

#### Camera Control
- **MovingCameraScene**: Scene where the camera can move around.
- **ZoomedScene**: Scene with a zoomed-in portion for detail.

---

## 4. Camera Classes

Camera classes control what is rendered and how.

- **Camera**: The base camera class for 2D rendering.
- **MovingCamera**: A camera that can move and pan.
- **ThreeDCamera**: Camera for 3D scenes.
- **MappingCamera**: Camera with custom coordinate mapping.
- **MultiCamera**: Multiple camera views.

---

## 5. Color Constants and Utilities

### Color Constants

Manim provides a comprehensive set of named colors:

**Grayscale Colors:**
- WHITE, BLACK
- GRAY_A/GRAY_B/GRAY_C/GRAY_D/GRAY_E (and GREY variants)
- LIGHT_GRAY, DARK_GRAY, LIGHTER_GRAY, DARKER_GRAY

**Primary Colors:**
- PURE_RED, PURE_GREEN, PURE_BLUE

**Color Variants (A=lightest, E=darkest):**
- RED, RED_A, RED_B, RED_C, RED_D, RED_E
- GREEN, GREEN_A, GREEN_B, GREEN_C, GREEN_D, GREEN_E
- BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E
- YELLOW, YELLOW_A, YELLOW_B, YELLOW_C, YELLOW_D, YELLOW_E
- TEAL, TEAL_A, TEAL_B, TEAL_C, TEAL_D, TEAL_E
- GOLD, GOLD_A, GOLD_B, GOLD_C, GOLD_D, GOLD_E
- PURPLE, PURPLE_A, PURPLE_B, PURPLE_C, PURPLE_D, PURPLE_E
- MAROON, MAROON_A, MAROON_B, MAROON_C, MAROON_D, MAROON_E

**Special Colors:**
- PINK, LIGHT_PINK, ORANGE, LIGHT_BROWN, DARK_BROWN, GRAY_BROWN

**Logo Colors:**
- LOGO_WHITE, LOGO_GREEN, LOGO_BLUE, LOGO_RED, LOGO_BLACK

### Color Utilities

- **ManimColor**: The main color class supporting multiple color spaces (RGB, HSV, HSL, hex).
  - `from_rgb()`: Create color from RGB values
  - `from_hex()`: Create color from hex string
  - `to_rgb()`: Convert to RGB
  - `to_hex()`: Convert to hex string
  - `contrasting()`: Get contrasting color (black or white)

---

## 6. Mathematical Constants and Functions

### Mathematical Constants

**Geometric Constants:**
- **ORIGIN**: np.array((0, 0, 0)) - The origin point
- **UP**: Unit vector in positive Y direction
- **DOWN**: Unit vector in negative Y direction
- **RIGHT**: Unit vector in positive X direction
- **LEFT**: Unit vector in negative X direction
- **IN**: Unit vector in negative Z direction (into screen)
- **OUT**: Unit vector in positive Z direction (out of screen)

**Axis Vectors:**
- **X_AXIS**: (1, 0, 0)
- **Y_AXIS**: (0, 1, 0)
- **Z_AXIS**: (0, 0, 1)

**Diagonal Directions:**
- **UL**: UP + LEFT (upper left)
- **UR**: UP + RIGHT (upper right)
- **DL**: DOWN + LEFT (lower left)
- **DR**: DOWN + RIGHT (lower right)

**Mathematical Constants:**
- **PI**: 3.14159... (π)
- **TAU**: 2π (tau, full rotation in radians)
- **DEGREES**: TAU/360 (conversion factor from degrees to radians)

**Default Sizes:**
- **DEFAULT_DOT_RADIUS**: 0.08
- **DEFAULT_SMALL_DOT_RADIUS**: 0.04
- **DEFAULT_DASH_LENGTH**: 0.05
- **DEFAULT_ARROW_TIP_LENGTH**: 0.35
- **DEFAULT_FONT_SIZE**: 48

**Buffers/Spacing:**
- **SMALL_BUFF**: 0.1
- **MED_SMALL_BUFF**: 0.25
- **MED_LARGE_BUFF**: 0.5
- **LARGE_BUFF**: 1.0
- **DEFAULT_MOBJECT_TO_EDGE_BUFFER**: MED_LARGE_BUFF
- **DEFAULT_MOBJECT_TO_MOBJECT_BUFFER**: MED_SMALL_BUFF

**Other Defaults:**
- **DEFAULT_STROKE_WIDTH**: 4
- **DEFAULT_POINT_DENSITY_2D**: 25
- **DEFAULT_POINT_DENSITY_1D**: 10

### Mathematical Functions (space_ops)

**Vector Operations:**
- `norm_squared(v)`: Square of vector magnitude
- `cross(v1, v2)`: Cross product
- `normalize(v)`: Normalize vector to unit length
- `normalize_along_axis(array, axis)`: Normalize along specific axis
- `angle_of_vector(vector)`: Get angle of vector
- `angle_between_vectors(v1, v2)`: Angle between two vectors
- `rotate_vector(vector, angle, axis)`: Rotate vector around axis
- `complex_to_R3(complex_num)`: Convert complex number to 3D point
- `R3_to_complex(point)`: Convert 3D point to complex number

**Geometric Operations:**
- `center_of_mass(points)`: Find center of mass of points
- `midpoint(p1, p2)`: Get midpoint between two points
- `line_intersection(p1, p2, p3, p4)`: Find intersection of two lines
- `get_unit_normal(v1, v2)`: Get unit normal to two vectors
- `compass_directions(n)`: Get n compass-like directions

**Matrix/Transform Operations:**
- `rotation_matrix(angle, axis)`: Create rotation matrix
- `rotation_matrix_transpose(angle, axis)`: Create transpose of rotation
- `rotation_about_z(angle)`: Rotation matrix about Z axis
- `quaternion_from_angle_axis(angle, axis)`: Convert angle-axis to quaternion
- `rotation_matrix_from_quaternion(quat)`: Convert quaternion to rotation matrix
- `z_to_vector(vector)`: Rotation matrix to align Z-axis with vector
- `thick_diagonal(dim, thickness)`: Create thick diagonal matrix

**Polygon Operations:**
- `regular_vertices(n, radius)`: Get vertices of regular n-gon
- `shoelace(x_y)`: Shoelace formula for polygon area
- `get_winding_number(points)`: Get winding number

### Simple Functions (simple_functions)

- `sigmoid(x)`: Logistic sigmoid function (1 / (1 + exp(-x)))
- `clip(a, min_a, max_a)`: Clip value to range
- `choose(n, k)`: Binomial coefficient (n choose k)
- `binary_search(function, target, lower, upper)`: Binary search for target value

---

## 7. Rate Functions (Animation Speed Curves)

Rate functions control the speed of animations over time.

### Standard Easing Functions

**Linear:**
- `linear`: Constant speed

**Smooth/Smoothstep:**
- `smooth`: Smooth acceleration/deceleration
- `smoothstep`: Smooth step function
- `smootherstep`: Even smoother step function
- `smoothererstep`: Ultra smooth step function
- `double_smooth`: Apply smooth twice

**Ease In/Out (various curves):**
- `rush_into`: Rush in at beginning
- `rush_from`: Rush from at end
- `slow_into`: Slow start
- `ease_in_sine`, `ease_out_sine`, `ease_in_out_sine`
- And many other ease variations...

**Special Functions:**
- `there_and_back`: Go forward then back
- `there_and_back_with_pause`: Go forward, pause, go back
- `running_start`: Start with motion already in progress
- `not_quite_there`: Reach 90% of target
- `wiggle`: Oscillate around target
- `lingering`: Linger at end
- `exponential_decay`: Exponential decay function
- `squish_rate_func`: Squish/compress animation

---

## 8. Configuration System

The global configuration system controls Manim's behavior.

### Accessing Config

```python
from manim import config
config['frame_height'] = 8.0
```

### Key Configuration Options

**Rendering:**
- `renderer`: 'cairo' or 'opengl'
- `frame_height`, `frame_width`: Scene dimensions
- `pixel_height`, `pixel_width`: Output resolution
- `frame_rate`: Animation frame rate (fps)

**Output:**
- `media_dir`: Directory for output files
- `video_dir`: Directory for video files
- `images_dir`: Directory for image files
- `tex_dir`: Directory for LaTeX files
- `write_to_movie`: Write video file
- `save_last_frame`: Save final frame as image
- `save_pngs`: Save all frames as PNGs
- `save_as_gif`: Save as GIF animation

**Quality:**
- `quality`: 'low_quality', 'medium_quality', 'high_quality', 'production_quality', 'fourk_quality'

**Background:**
- `background_color`: Scene background color
- `background_opacity`: Background transparency

**Logging:**
- `verbosity`: Logging level

### Temporary Config Changes

```python
from manim import config, tempconfig

with tempconfig({"frame_height": 10.0}):
    # Config is temporarily changed
    pass
# Config is restored
```

---

## 9. Utility Functions and Helpers

### Iterables Module

List and sequence utility functions:
- `adjacent_n_tuples(objects, n)`: Get adjacent n-tuples from sequence
- `adjacent_pairs(objects)`: Get adjacent pairs
- `all_elements_are_instances(iterable, Class)`: Check if all are instances
- `batch_by_property(iterable, prop_func)`: Batch by property
- `concatenate_lists(*lists)`: Concatenate multiple lists
- `listify(obj)`: Convert to list
- `make_even(list)`: Ensure list has even length
- `remove_nones(sequence)`: Remove None values
- `resize_array(array, length)`: Resize numpy array
- `stretch_array_to_length(array, length)`: Stretch array to length
- `resize_with_interpolation(array, length)`: Resize with interpolation

### File Operations (file_ops)
- `open_file(path)`: Open file with default application
- `find_file(filename)`: Find file in Manim directories

### Paths Module
- Path manipulation utilities for media, video, image directories

### Bezier Module
Bezier curve operations for smooth paths.

---

## 10. Enumerations

### Line Joint Types (LineJointType)
- `AUTO`: Automatic joint selection
- `ROUND`: Rounded joints
- `BEVEL`: Beveled joints
- `MITER`: Mitered joints

### Cap Style Types (CapStyleType)
- `AUTO`: Automatic cap style
- `ROUND`: Rounded caps
- `BUTT`: Butted (square) caps
- `SQUARE`: Square caps

### Renderer Type (RendererType)
- `CAIRO`: Cairo backend renderer
- `OPENGL`: OpenGL renderer

### Font Styles
- `NORMAL`: Normal font
- `ITALIC`: Italic font
- `OBLIQUE`: Oblique font
- `BOLD`: Bold font
- `THIN`, `ULTRALIGHT`, `LIGHT`, `SEMILIGHT`: Light weights
- `BOOK`, `MEDIUM`: Medium weights
- `SEMIBOLD`, `ULTRABOLD`, `HEAVY`, `ULTRAHEAVY`: Heavy weights

---

## 11. Quality Presets

Video quality configurations available:
- `low_quality`: 854x480 @ 15fps
- `medium_quality`: 1280x720 @ 30fps
- `high_quality`: 1920x1080 @ 60fps
- `production_quality`: 2560x1440 @ 60fps
- `fourk_quality`: 3840x2160 @ 60fps
- `example_quality`: 854x480 @ 30fps

---

## 12. Import Examples

### Basic Imports
```python
from manim import *
# This imports all commonly used classes and constants

from manim import Scene, Circle, Square, Create, FadeIn
from manim import RED, GREEN, BLUE, WHITE, BLACK
from manim import UP, DOWN, LEFT, RIGHT, ORIGIN
from manim import config
```

### Specific Module Imports
```python
from manim import rate_functions
from manim.utils import color
from manim.utils.space_ops import normalize, rotate_vector
```

---

## 13. Common Usage Patterns

### Basic Scene Structure
```python
from manim import *

class MyScene(Scene):
    def construct(self):
        circle = Circle()
        self.add(circle)
        self.play(Create(circle))
        self.play(FadeOut(circle))
```

### Working with Colors
```python
from manim import *

circle = Circle(color=RED, fill_color=BLUE, fill_opacity=0.5)
circle.set_color(GREEN)
circle.set_fill(YELLOW, opacity=0.8)
```

### Animation Sequences
```python
self.play(Create(obj1))
self.wait(1)
self.play(Transform(obj1, obj2))
self.wait(0.5)
self.play(FadeOut(obj2), run_time=2)
```

### Value Tracking and Updaters
```python
value_tracker = ValueTracker(0)
decimal = DecimalNumber(value_tracker.get_value())
decimal.add_updater(lambda m: m.set_value(value_tracker.get_value()))

self.add(decimal, value_tracker)
self.play(value_tracker.animate.set_value(10), run_time=2)
```

---

## Summary Table

| Category | Count | Key Examples |
|----------|-------|--------------|
| Animation Classes | 60+ | Create, Transform, FadeIn, Indicate, Rotate |
| Mobject Classes | 100+ | Circle, Square, Line, Text, Axes, Sphere |
| Scene Types | 6 | Scene, ThreeDScene, MovingCameraScene |
| Camera Types | 5 | Camera, MovingCamera, ThreeDCamera |
| Colors | 100+ | RED, GREEN, BLUE, and variants |
| Rate Functions | 30+ | linear, smooth, ease_in_sine, there_and_back |
| Math Functions | 50+ | normalize, rotate_vector, cross, center_of_mass |

This comprehensive documentation covers the entire public API of Manim, providing users with a complete reference for creating mathematical animations.
