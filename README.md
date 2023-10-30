# Surface plotting for FEniCS

 <p align="center">
    <img src="https://github.com/MiroK/surf-plot/blob/master/doc/logo.png">
  </p>

Simple plotting of X-d functions on (X-1)-d manifolds. The underlying idea
is to use the order 0 Discontinuous Lagrange Trace space as an
intermediate representation of data before plotting it as a piecewise constant
function on the manifold.

## Dependencies
- FEniCS 2019.1.0 or newer (it will work with 2017 if MPI stuff is adjusted, e.g. `MPI.comm_self`
- VTK (there is Ubuntu package for it)
- (Py)Evtk see [here](https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python) or [here](https://github.com/paulo-herrera/PyEVTK).
Note that depending on how the package is installed you might need to change the imports
in **vtk_io.py** from `import evtk` to `import pyevtk`

## Installation
Put `surf_plot` on PYTHONPATH

## Limitations
- [ ] I only needed this for scalars so atm only scalar fields are supported
- [ ] The API could certainly be improved
