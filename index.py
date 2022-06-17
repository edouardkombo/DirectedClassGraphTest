import numpy as np
from DirectedClassGraph import DirectedClassGraph

# build a random graph
num_vertices = 1000
num_edges = 100000

vertices  = list(range(num_vertices))
from_node = np.random.randint(num_vertices, size=(num_edges,))
to_node = np.random.randint(num_vertices, size=(num_edges,))
edges = list(zip(from_node, to_node))

# plot out degree
sdg = DirectedClassGraph(vertices, edges)
sdg.plot_outdegrees()

# display info
sdg.print_num_edges()
sdg.print_num_vertices()

# export to hdf5
sdg.save_to_hdf5('my_graph.hd5', 'graph1')

# load from hdf5
sdg2 = DirectedClassGraph.fromhdf5('my_graph.hd5', 'graph1')
sdg2.plot_outdegrees()
