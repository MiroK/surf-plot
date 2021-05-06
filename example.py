from dolfin import *

from vtk_io import DltWriter
from dlt_embedding import P0surf_to_DLT0_map
import numpy as np

mesh = UnitCubeMesh(6, 6, 6)
facet_f = MeshFunction('size_t', mesh, 2, 0)
CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 1)
CompiledSubDomain('near(x[1], 0.5)').mark(facet_f, 2)
CompiledSubDomain('near(x[2], 0.5)').mark(facet_f, 3)
    
cell_f = MeshFunction('size_t', mesh, 3, 0)
CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS').mark(cell_f, 1)

dx = Measure('dx', domain=mesh, subdomain_data=cell_f)        
ds = Measure('ds', domain=mesh, subdomain_data=facet_f)
dS = Measure('dS', domain=mesh, subdomain_data=facet_f)

surface_tags = (1, 2, 3)

surface_mesh, L, cell2Ldofs = P0surf_to_DLT0_map(facet_f, tags=surface_tags)    

V = FunctionSpace(mesh, 'CG', 1)
uh = interpolate(Expression('x[0]+x[1]+x[2]', degree=1), V)
    
V = FunctionSpace(mesh, 'DG', 0)  
v = TestFunction(V)
# To avoid including dx we shall refer to restricted quantities by the
# value of the subdomain_indicator function
subdomain_tag = Function(V)
hK = CellVolume(mesh)
# Project coefs to P0 based on subdomain
assemble((1/hK)*inner(v, Constant(1))*dx(1) + (1/hK)*inner(v, Constant(2))*dx(2), subdomain_tag.vector())
    
big_side = lambda u, K=subdomain_tag: conditional(gt(K('+'), K('-')), u('+'), u('-'))
small_side = lambda u, K=subdomain_tag: conditional(le(K('+'), K('-')), u('+'), u('-'))


# So finally for the projection
dl = TestFunction(L)
fK = avg(FacetArea(mesh))
# To get the quantity on the surface
vh = Function(L)

fK = avg(FacetArea(mesh))
dS0 = dS.reconstruct(metadata={'quadrature_degree': 0})

assemble(sum((1/fK)*inner(big_side(uh), avg(dl))*dS0(tag) for tag in surface_tags), tensor=vh.vector())
as_backend_type(vh.vector()).update_ghost_values()

cell_data = {'uh': vh.vector().get_local()[cell2Ldofs]}

dlt = DltWriter('testing/foo', mesh, surface_mesh)
with dlt as output:
    output.write(cell_data, t=0)

# Check the data
# Our projection is equivalent to evaluating the uh function (restriction)
# in the midpoint of the facet
midpoints = np.mean(surface_mesh.coordinates()[surface_mesh.cells()], axis=1)
true = np.sum(midpoints, axis=1)

mine = vh.vector().get_local()[cell2Ldofs]

assert np.linalg.norm(true - mine) < 1E-10
