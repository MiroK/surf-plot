from evtk.hl import unstructuredGridToVTK
from evtk.vtk import VtkTriangle, VtkGroup
import numpy as np
import os


pvtu_code = '''<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="0.1">
<PUnstructuredGrid GhostLevel="0">
<PPoints>
<PDataArray type="Float64" NumberOfComponents="3" />
</PPoints>
<PCellData>
<PDataArray type="UInt32" Name="connectivity" />
<PDataArray type="UInt32" Name="offsets" />
<PDataArray type="UInt8" Name="types" />
</PCellData>
<PCellData>
%(cell_data)s
</PCellData>
%(pieces)s
</PUnstructuredGrid>
</VTKFile>
'''

celldata_code = '<PDataArray type="Float64" Name="%(f)s" NumberOfComponents="0" />'


class DltWriter(object):
    '''
    VTK writer for scalar order 0 discontinuous Lagrange trace functions.
    '''
    def __init__(self, path, mesh, submesh):
        '''Prebuild and then only receive data'''
        comm = mesh.mpi_comm()
        rank = comm.rank
        size = comm.size

        dirname, basaname = os.path.dirname(path), os.path.basename(path)
        if dirname:
            not os.path.exists(dirname) and comm.rank == 0 and os.mkdir(dirname)
        
        local_has_piece = np.zeros(comm.size, dtype=bool)

        if submesh.num_cells() == 0:
            self.write_vtu_piece = lambda data, counter, path=path, rank=rank, size=size:(
                "%s_p%d_of%d_%06d.vtu" % (path, rank, size, counter)
            )
        else:
            local_has_piece[comm.rank] = True
            
            x, y, z = map(np.array, submesh.coordinates().T)
            cells = submesh.cells()
            ncells, nvertices_cell = cells.shape
            assert nvertices_cell == 3  
        
            connectivity = cells.flatten()
            offsets = np.cumsum(np.repeat(3, ncells))            
            cell_types = VtkTriangle.tid*np.ones(ncells)

            self.write_vtu_piece = lambda data, counter, path=path, rank=rank, size=size:(
                unstructuredGridToVTK("%s_p%d_of%d_%06d" % (path, rank, size, counter),
                                      x, y, z,
                                      connectivity,
                                      offsets,
                                      cell_types,
                                      cellData=data)
            )
        # Root will write the group file
        global_has_piece = sum(comm.allgather(local_has_piece),
                               np.zeros_like(local_has_piece))
        self.has_piece,  = np.where(global_has_piece)
        # Only other piece to remember is for parallel writeing on root
        self.counter = 0  # Of times write_vtu_piece was called
        self.path = path
        self.world_rank = comm.rank
        self.world_size = comm.size

        # Also group file goes into the directory
        self.world_rank == 0 and setattr(self, 'group', VtkGroup(path))

    def __enter__(self):
        return self
    
    def write(self, data, t):
        '''Write a new piece'''
        # Each process writes the vtu file - goes possibly to directory
        group_path = self.write_vtu_piece(data=data, counter=self.counter)

        # Root write one pvtu file
        # A pvtu file goes into the directory but but the pieces should use
        # only baname to point to vtu file
        if self.world_size > 1:
            group_path = '%s_of%d_%06d.pvtu' % (self.path, self.world_size, self.counter)

            cell_data_vtk = '\n'.join([celldata_code % {'f': key} for key in data])

            if self.world_rank == 0:
                with open(group_path, 'w') as group_file:
                    group_file.write(pvtu_code %
                        {'cell_data': cell_data_vtk,
                         'pieces': '\n'.join(
                             ['<Piece Source="%s_p%d_of%d_%06d.vtu" />' % (os.path.basename(self.path),
                                                                           rank,
                                                                           self.world_size,
                                                                           self.counter)
                              for rank in self.has_piece])
                        }
                    )

        self.counter += 1  # Of times write_vtu_piece was called
                                     
        # Update group file; refer to group item as
        self.world_rank == 0 and self.group.addFile(filepath=group_path, sim_time=t)
            
    def __exit__(self, exc_type, exc_value, exc_traceback):
        '''Complete the pvd file'''
        self.world_rank == 0 and self.group.save()
