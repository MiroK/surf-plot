# Plotting P1 functions

from dolfin import *

from surf_plot.vtk_io import DltWriter
from surf_plot.dlt_embedding import P0surf_to_DLT0_map
import numpy as np

mesh = UnitCubeMesh(6, 6, 6)
facet_f = MeshFunction('size_t', mesh, 2, 0)
# The manifold can use several markers
CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 1)
CompiledSubDomain('near(x[1], 0.5)').mark(facet_f, 2)
CompiledSubDomain('near(x[2], 0.5)').mark(facet_f, 3)
# We will use projection on the tagged manifolds so the measure must be aware
# of the surfaces
ds = Measure('ds', domain=mesh, subdomain_data=facet_f)
dS = Measure('dS', domain=mesh, subdomain_data=facet_f)

surface_tags = (1, 2, 3)
# What were after is filling in a P0 function on the surface mesh using
# L = FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0) and the dof
# mapping. This would represent surface of the neuron
surface_mesh, L, cell2Ldofs = P0surf_to_DLT0_map(facet_f, tags=surface_tags)    

f = Expression('x[0]+x[1]+time*x[2]', degree=1, time=0)
# So let's make some data
V = FunctionSpace(mesh, 'DG', 1)
uh = interpolate(f, V) # This is EMI potential
    
# Now we project
dl = TestFunction(L)
fK = avg(FacetArea(mesh))
# To get the quantity on the surface
vh = Function(L)

fK = FacetArea(mesh)
dS0 = dS.reconstruct(metadata={'quadrature_degree': 0})
ds0 = ds.reconstruct(metadata={'quadrature_degree': 0})
# NOTE: here the scaling by 1/fK and the quadrature rule means that we get
# the DLT0 value by evaluating the function in the facet midpoint. Also, we
# have data (uh) in continuous space so avg(uh) = uh on the surface
L_form = sum((1/avg(fK))*inner(avg(uh), avg(dl))*dS0(tag) + (1/fK)*inner(uh, dl)*ds0(tag)
             for tag in surface_tags)
assemble(L_form, tensor=vh.vector())
as_backend_type(vh.vector()).update_ghost_values()
# At this point vh has the quantitiy of interest. Iguess it typically is
# some restriction, see `example_P0.py` for how to get them

# For plotting we always want grab subset of DLT dofs
values = vh.vector().get_local()[cell2Ldofs]

# And dump - note that we can have several quantities (e.g. neg_ih) in the
# same file
cell_data = {'uh': values, 'neg_uh': values}

dlt = DltWriter(f'testing/cux_world{mesh.mpi_comm().size}', mesh, surface_mesh)
with dlt as output:
    output.write(cell_data, t=0)

    # Some more time steps
    for time in (1, 2, 3, 4):
        f.time = time
        uh.interpolate(f)
        
        assemble(L_form, tensor=vh.vector())
        as_backend_type(vh.vector()).update_ghost_values()

        # Now we build the data for P0 function on the mesh
        values[:] = vh.vector().get_local()[cell2Ldofs]
        print(np.linalg.norm(values))
        output.write(cell_data, t=time)
