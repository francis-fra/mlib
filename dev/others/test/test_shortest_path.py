import sys
srcDir = '/home/fra/FraDir/learn/Learnpy/Mypy/graph'

sys.path.append(srcDir)

from imp import reload

import pygraph
reload(pygraph)
from pygraph import *

#----------------------------------------------------------------
g = DiGraph()
# add edges    
g.addEdge(0,1,5)
g.addEdge(0,5,2)
g.addEdge(1,2,4)
g.addEdge(2,3,9)
g.addEdge(3,4,7)
g.addEdge(3,5,3)
g.addEdge(4,0,1)
g.addEdge(5,4,8)
g.addEdge(5,2,1)

# plt.figure()
g.draw_graph()

g.shortest_path(0, 3)
g.shortest_path(0, 4)
sys.maxsize
# shortest path

shortest_path(g, 0, 3)
shortest_path(g, 0, 4)
shortest_path(g, 0, 1)
shortest_path(g, 0, 2)

shortest_path(g, 3, 1)
shortest_path(g, 3, 2)
shortest_path(g, 4, 2)

g.clear_all_data()
g.get_all_data()

path
path.reverse()
path
g.getVertices()

#----------------------------------------------------------------
# DEBUG

vertexDict = g.getVertices()
vertexDict
vertexDict[3].data
vertexDict[0].data
vertexDict[5].data
vertexDict[4].data


# vertexDict[0]
# V1 = SortableVertex(vertexDict[0], 10)
# V1.get_distance()
# V2 = SortableVertex(vertexDict[1], 3)
# V1 < V2

import queue
q = queue.PriorityQueue()

q.put(V1)
q.put(V2)
q.queue

dir(q)
q.qsize()
q.get()

V1.V.neighbours()

g.neighbours(V1.V.id)

shortest_path(g, 0)