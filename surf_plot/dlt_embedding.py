from dolfin import parameters, FunctionSpace, MPI
from surf_plot.embedded_mesh import EmbeddedMesh
import numpy as np


parameters['ghost_mode'] = 'shared_vertex'


def P0surf_to_DLT0_map(facet_f, tags):
    '''
    Compute mapping for using DLT0 functions to set P0 functions on the 
    embedded mesh of entities where facet_f == tags
    '''
    # The idea here is that we only want to include in the submesh the
    # facets which this process can write to. This is determined by auxiliary
    # DLT0 space, because we want to represent data by P0 on the neuron
    # surface mesh
    mesh = facet_f.mesh()
    L = FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)    
    
    tdim = mesh.topology().dim()
    dm = L.dofmap()
    # Mapping of facets to degrees of freedom ...
    facet2Ldofs = np.array(dm.entity_closure_dofs(mesh, tdim-1))
    # ... we are only interested in not-ghosts
    facet2Ldofs, ghosts = facet2Ldofs[:mesh.topology().ghost_offset(tdim-1)], facet2Ldofs[mesh.topology().ghost_offset(tdim-1):]
    # And further we want to filter by ownership
    first, last = dm.ownership_range()
    global_dofs = np.fromiter(map(dm.local_to_global_index, facet2Ldofs), dtype=facet2Ldofs.dtype)
    owned, = np.where(np.logical_and(global_dofs >= first, global_dofs < last))
    # So when we will build the mesh out of marked entities
    ignore = set(range(mesh.num_entities(tdim-1))) - set(owned)
    # Now we want to create mesh for the neuron surface where each cell
    # is for one facet owned by DLT
    comm = MPI.comm_self  # The mesh is always local
    surface_mesh = EmbeddedMesh(facet_f, tags, comm, ignored_entities=ignore)
    ncells = surface_mesh.num_cells()
    # For io, whatever we want to evalute the idea is to first interpolate
    # it to DLT and then assign as if to P0 function on the mesh using
    if ncells > 0:
        c2f = surface_mesh.parent_entity_map[mesh.id()][tdim-1]    
        cell2Ldofs = np.array([facet2Ldofs[c2f[c]] for c in range(ncells)])
    else:
        cell2Ldofs = []

    return surface_mesh, L, cell2Ldofs
