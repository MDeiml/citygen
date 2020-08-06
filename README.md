# Citygen

This is a project to create a procedurally generated city, which expands in all 3 dimensions.
The city should appear as if it grew naturally, which means, that it should contain no uniform structure, like a grid of walkways or any other type of orginization.
Rather the houses and walkways should be placed seemingly randomly.

This creates some challenges, as there is for example no straightforward algorithm to partition an infinite 3D world into cuboids (houses), and which can run in finite time to determine the partition of a subvolume or "chunk".

## Procedural generation

The generation process is subdivided into steps to create on type of structure at a time:

1. Walkways
    1. Crossing positions
    2. Connecting ways
2. House
    1. Rectangular bounds
    2. Details (windows, doors, ...)

For each step first a random solution is chosen, which might not conform to all necessary restrictions.
For example the bounds for neighbouring houses could be chosen in such a ways that they intersect.
To limit the complexity the world is subdivided into a grid of "cells" and only one structure is generated for each cell.
The restrictions are then solved by taking into account neighbouring cells.
In the solving algorithm between two neighbours there is always a priority as to which cell should be changed.
This way restrictions can be solved wihout creating new ones and the final result of structures of one cell depends only on calculations in a finite number of neighbouring cells.

Most of the procedural generation can be run in a compute shader to use the graphics cards capabilities to enable real time generation.

### Walkways

First a random position inside each cell and it is chosen if a cell should try to align itself with it positive X- and Z-axis neighbours or the neighbours above or below those cells.
This means each cell has the following properties:

| Property  | Possible values          |
| --------- | ------------------------ |
| `offset`  | `[(0, 0, 0), cell_size)` |
| `align_x` | `no_align, +Y, -Y, 0`    |
| `align_z` | `no_align, +Y, -Y, 0`    |

In the second step each cell tries to align the x and y components of its offset accoring to `align_x` and `align_y`.
For this it copies the previous `y` component (or `x` component in the case of `align_z`) of the coresponding neighbouring cell into its own offset.
This step is repeated once more, to ensure that it is common enough for a "straight" crossing to be created.

With the crossing positions now calculated a path is created between each pair of neighbouring cell where this is possible.
A priority should be chosen to ensure no confilicting paths are chosen.

### Houses

For each cell a random cuboid is chosen, which might reach into the cells positive axis neighbours.
This means each cell has the properties:

| Property | Possible values          |
| -------- | ------------------------ |
| `offset` | `[(0, 0, 0), cell_size)` |
| `size`   | `[min_size, cell_size)`  |

It should then be shrunken so it does not intersect any neighbouring houses or paths.
As before a priority should be chosen as to which of two neightbouring houses should be shrunken.

## Artistic Design

### Rendering

This project will use Sparse Voxel Octree (SVO) raytracing for rendering, which enables many intersting lighting models.
It uses Directed Acyclic Graphs (DAGs) instead of SVOs to save graphics memory, as the world can easily be subdivided into repeating "blocks".

### Architecture

The city should appear as if the houses are "self-built" meaning that houses should contain simple materials and feature "unprofessional" decorations such as open wires or pipes.
Its inhabitants' technology should be ahead of the current state, meaning that there are some futuristic objects and machines, but technology should be used rather primitively, as if there was no deeper understanding into how it works.
