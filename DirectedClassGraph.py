from typing import List, Tuple, Union, Optional
import h5py
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


HDF5_VERTEX_NAMES = 'vertices_names'
HDF5_ADJ_INDICES = 'adj_indices'
HDF5_ADJ_INDPTR = 'adj_indptr'
HDF5_NUM_VERTICES = 'num_vertices'
HDF5_NUM_EDGES = 'num_edges'


class DirectedClassGraph():

    def __init__(self,
                 vertices: List[Union[int, str]],
                 edges: Optional[List[Tuple[Union[int, str]]]] = None,
                 adjacency: Optional[Tuple[ArrayLike, ArrayLike]] = None):

        if len(set(vertices)) != len(vertices):
            raise ValueError("Duplicate vertices not supported")

        self.vertices = vertices
        self.num_vertices = len(vertices)

        if (adjacency is None) == (edges is None):
            raise ValueError("Either provide edges or adjacency matrix")

        if adjacency is not None:
            # adjacency is given as ranks (idx in the list of unique vertices)
            indices = np.array(adjacency[0])
            indptr = np.array(adjacency[1])
            num_edges = indices.size
            self.adjacency = csr_matrix((np.ones((num_edges), 'bool'), indices, indptr),
                                        shape=(self.num_vertices, self.num_vertices))
        elif edges is not None:
            # need to convert vertices in the given edges to ranks
            vertices_to_rank = {v: i for i, v in enumerate(self.vertices)}

            num_edges = len(edges)
            rank_in = np.zeros((num_edges), 'int')
            rank_out = np.zeros((num_edges), 'int')
            for i, (in_vertex, out_vertex) in enumerate(edges):  # type: ignore
                rank_in[i] = vertices_to_rank[in_vertex]  # type: ignore
                rank_out[i] = vertices_to_rank[out_vertex]  # type: ignore

            # use bool to save space and avoid duplication
            self.adjacency = csr_matrix((np.ones((num_edges), 'bool'), (rank_in, rank_out)),
                                        shape=(self.num_vertices, self.num_vertices))

        self.num_edges = self.adjacency.sum()

    def print_num_edges(self) -> int:
        print(f"The graph has {self.num_edges} (unique) edges.")
        return self.num_edges

    def print_num_vertices(self) -> int:
        print(f"The graph has {self.num_vertices} (unique) vertices.")
        return self.num_vertices

    def save_to_hdf5(self, hdf5_fname: str, group_name: str) -> None:
        """ Save graph to hdf5 file

        Args:
            hdf5_fname: file name of hdf5 data base
            group_name: hdf5 group name to store the graph in
        """

        with h5py.File(hdf5_fname, 'w') as file:
            grp = file.create_group(group_name)
            grp.create_dataset(HDF5_VERTEX_NAMES, data=self.vertices)
            grp.create_dataset(HDF5_ADJ_INDICES, data=self.adjacency.indices)
            grp.create_dataset(HDF5_ADJ_INDPTR, data=self.adjacency.indptr)

            # save some info
            grp.attrs[HDF5_NUM_VERTICES] = self.num_vertices
            grp.attrs[HDF5_NUM_EDGES] = self.num_edges

    def compute_outdegrees(self) -> np.ndarray:
        """ Computes the out degrees for each vertex.

        Returns:
            vector of out degrees
        """
        return self.adjacency.sum(axis=1)

    def plot_outdegrees(self, fname: Optional[str] = None, num_bins: Optional[int] = None) -> None:
        """ Plots a histogram of out degrees of the graph and saves it as PNG file.

        Args:
            fname: if give, saves the plot in an PNG file
            num_bins: number of bins. Estimated automatically if not provided
        """

        plt.figure()
        out_degrees = self.compute_outdegrees()

        bin_range = None
        if num_bins is None:
            bin_range = [out_degrees.min(), out_degrees.max()]
            num_bins = bin_range[1] - bin_range[0]  # type: ignore

        plt.hist(out_degrees, num_bins, range=bin_range)
        plt.xlabel('Out-degree')
        plt.ylabel('# count')
        plt.title(f'Out-degree histogram of graph \n({self.num_vertices} vertices, '
                  f'{self.num_edges} edges)')

        if fname is None:
            plt.ion()
            plt.show()
        else:
            plt.savefig(fname, format='png')

    @classmethod
    def fromhdf5(cls, hdf5_fname: str, group_name: str) -> 'DirectedClassGraph':

        with h5py.File(hdf5_fname, 'r') as file:

            if group_name not in file:
                raise ValueError(f"Cannot find group name '{group_name}' in hdf5 file!")

            grp = file[group_name]

            if HDF5_VERTEX_NAMES not in grp:
                raise ValueError("Cannot find graph in hdf5 group!")

            vertices = grp[HDF5_VERTEX_NAMES][:].tolist()
            indices = grp[HDF5_ADJ_INDICES][:]
            indptr = grp[HDF5_ADJ_INDPTR][:]

        if len(vertices) and isinstance(vertices[0], bytes):
            # simple decode for strings
            vertices = [v.decode() for v in vertices]

        return cls(vertices, adjacency=(indices, indptr))
