from surf_plot.make_mesh_cpp import make_mesh
import dolfin as df
import numpy as np


class EmbeddedMesh(df.Mesh):
    '''
    Construct a mesh of marked entities in marking_function.
    The output is the mesh with cell function which inherited the markers. 
    and an antribute `parent_entity_map` which is dict with a map of new 
    mesh vertices to the old ones, and new mesh cells to the old mesh entities.
    Having several maps in the dict is useful for mortaring.
    '''
    def __init__(self, marking_function, markers, comm, ignored_entities=None):
        if not isinstance(markers, (list, tuple)): markers = [markers]
        
        base_mesh = marking_function.mesh()
        assert base_mesh.topology().dim() > marking_function.dim()

        gdim = base_mesh.geometry().dim()
        tdim = marking_function.dim()
        assert tdim > 0, 'No Embedded mesh from vertices'

        assert markers, markers

        # Otherwise the mesh needs to by build from scratch
        _, e2v = (base_mesh.init(tdim, 0), base_mesh.topology()(tdim, 0))
        entity_values = marking_function.array()
        
        if ignored_entities is not None:
            ignored_entities = set(ignored_entities)
            colorings = [list(set(np.where(entity_values == tag)[0])-ignored_entities) for tag in markers]
        else:
            colorings = [np.where(entity_values == tag)[0] for tag in markers]
        # Represent the entities as their vertices
        tagged_entities = np.hstack(colorings)

        # Nothing to do
        if len(tagged_entities) == 0:
            # With acquired data build the mesh
            df.Mesh.__init__(self, comm)
    
            mesh_key = marking_function.mesh().id()
            self.parent_entity_map = {mesh_key: {0: {}, tdim: {}}}
            return None

        tagged_entities_v = np.array([e2v(e) for e in tagged_entities], dtype='uintp')
        # Unique vertices that make them up are vertices of our mesh
        tagged_vertices = np.unique(tagged_entities_v.flatten())
        # Representing the entities in the numbering of the new mesh will
        # give us the cell makeup
        mapping = dict(zip(tagged_vertices, range(len(tagged_vertices))))
        # So these are our new cells
        tagged_entities_v.ravel()[:] = np.fromiter((mapping[v] for v in tagged_entities_v.flat),
                                                   dtype='uintp')
        
        # With acquired data build the mesh
        df.Mesh.__init__(self, comm)
        # Fill
        vertex_coordinates = base_mesh.coordinates()[tagged_vertices]
        make_mesh(coordinates=vertex_coordinates, cells=tagged_entities_v, tdim=tdim, gdim=gdim,
                  mesh=self)
        # The entity mapping attribute
        mesh_key = marking_function.mesh().id()
        self.parent_entity_map = {mesh_key: {0: dict(enumerate(tagged_vertices)),
                                             tdim: dict(enumerate(tagged_entities))}}

        f = df.MeshFunction('size_t', self, tdim, 0)
        # Finally the inherited marking function. We colored sequentially so
        if len(markers) > 1:
            f_ = f.array()            
            offsets = np.cumsum(np.r_[0, list(map(len, colorings))])
            for i, marker in enumerate(markers):
                f_[offsets[i]:offsets[i+1]] = marker
        else:
            f.set_all(markers[0])

        self.marking_function = f
        # Declare which tagged cells are found
        self.tagged_cells = set(markers)

# --------------------------------------------------------------------

if __name__ == '__main__':
    # This is for 2019 and up
    mesh = df.UnitSquareMesh(32, 32)
    facet_f = df.MeshFunction('size_t', mesh, 1, 0)
    code = ' || '.join(['(near(x[0], 0.25) && (0.25-tol < x[1]) && (x[1] < 0.75+tol))',
                        '(near(x[0], 0.75) && (0.25-tol < x[1]) && (x[1] < 0.75+tol))',
                        '(near(x[1], 0.25) && (0.25-tol < x[0]) && (x[0] < 0.75+tol))',
                        '(near(x[1], 0.75) && (0.25-tol < x[0]) && (x[0] < 0.75+tol))'])
    df.CompiledSubDomain(code, tol=1E-10).mark(facet_f, 1)

    emesh = EmbeddedMesh(facet_f, markers=(1, ), comm=df.MPI.comm_self)
