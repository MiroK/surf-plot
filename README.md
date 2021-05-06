# Surface plotting for FEniCS

Simple plotting of X-d functions on (X-1)-d manifolds. The underlying idea
is to use the order 0 Discontinuous Lagrange Trace space as an
intermediate representation of data before plotting it as a piecewise constant
function on the manifold.

## Dependencies
- VTK (there is Ubuntu package for it)
- (Py)Evtk see [here](https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python). Note
that depending on how the package is installed you might need to change the imports
in **vtk_io.py** from `import evtk` to `import pyevtk`

## Limitations
- [ ] I only needed this for scalars so atm only scalar fields are supported
- [ ] The API could certainly be improved