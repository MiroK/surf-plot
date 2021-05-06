# Plotting P0 function (mostly used to illustrate how to do restrictions)

from dolfin import *

from surf_plot.vtk_io import DltWriter
from surf_plot.dlt_embedding import P0surf_to_DLT0_map
import numpy as np

mesh = UnitCubeMesh(6, 6, 6)
# Here we will have two marked subdomains ...
cell_f = MeshFunction('size_t', mesh, 3, 1)
CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS').mark(cell_f, 2)
# ... with their interface
facet_f = MeshFunction('size_t', mesh, 2, 0)
CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 1)
    

ds = Measure('ds', domain=mesh, subdomain_data=facet_f)
dS = Measure('dS', domain=mesh, subdomain_data=facet_f)

surface_tags = (1, )

surface_mesh, L, cell2Ldofs = P0surf_to_DLT0_map(facet_f, tags=surface_tags)    
# Let there be som discontinuous data ...
V = FunctionSpace(mesh, 'DG', 0)
uh = interpolate(Expression('x[0] > 0.5 - DOLFIN_EPS ? 1: 2', degree=0), V)
# ... and we want to represent parts from both subdomains sharing the interface

# To get the restriction as we want we build a P0(mesh) tagging function 
dx = Measure('dx', domain=mesh, subdomain_data=cell_f)

V = FunctionSpace(mesh, 'DG', 0)  
v = TestFunction(V)

subdomain_tag = Function(V)
hK = CellVolume(mesh)
# Project coefs to P0 based on subdomain
assemble((1/hK)*inner(v, Constant(1))*dx(1) + (1/hK)*inner(v, Constant(2))*dx(2), subdomain_tag.vector())
# With the cell taggging function we will pick sides based on the marker value
big_side = lambda u, K=subdomain_tag: conditional(gt(K('+'), K('-')), u('+'), u('-'))
small_side = lambda u, K=subdomain_tag: conditional(le(K('+'), K('-')), u('+'), u('-'))

# So finally for the projection of P0 data
dl = TestFunction(L)
fK = avg(FacetArea(mesh))
# To get the quantity on the surface
vh = Function(L)

fK = FacetArea(mesh)
dS0 = dS.reconstruct(metadata={'quadrature_degree': 0})
ds0 = ds.reconstruct(metadata={'quadrature_degree': 0})

for truth, side in enumerate((big_side, small_side), 1):
    # NOTE: that DLT is "continuous" on the facet, avg is just to shout up FFC
    assemble(sum((1/avg(fK))*inner(side(uh), avg(dl))*dS0(tag) + (1/fK)*inner(uh, dl)*ds0(tag)
                 for tag in surface_tags), tensor=vh.vector())
    as_backend_type(vh.vector()).update_ghost_values()

    values = vh.vector().get_local()[cell2Ldofs]
    cell_data = {'uh': values}

    dlt = DltWriter('testing/bar', mesh, surface_mesh)
    with dlt as output:
        output.write(cell_data, t=0)

    # Check the data
    true = truth*np.ones_like(values)
    assert np.linalg.norm(true - values) < 1E-12
