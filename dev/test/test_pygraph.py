import sys
srcDir = '/home/fra/FraDir/learn/Learnpy/Mypy/graph'

sys.path.append(srcDir)

#----------------------------------------------------------------
# bfs
#----------------------------------------------------------------
from imp import reload

import pygraph
reload(pygraph)
from pygraph import *

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

g.diameter()
g.find_isolated_nodes()
g.order()
g.size()
g.density()
g.is_connected()


def printID(v):
    "print ID"
    print("Visiting {}".format(v.id))

def setState(v):
    "setState"
    v.set_data("state", 1)
    
T = Task(g, actionList=[printID, setState])
T = Task(g, actionList=[])
metaData = bfs(T,5)
metaData = bfs(T,0)
bfs(T,1)
bfs(T,0)
bfs(T,5)

metaData
parse_path(metaData)
g.get_all_data()

metaData = dfs(T, 5)  
metaData = dfs(T, 3)  
metaData = dfs(T, 0)  
metaData

g.is_connected()

#----------------------------------------------------------------
# DEBUG
#----------------------------------------------------------------



  
# g.vertList[0].getConnections()
a = [x for x in g.neighbours(1)][0]
type(a)
a.id

a.id = 9

list(g.neighbours(0))
list(g.neighbours(1))

#----------------------------------------------------------------
from pygraph import *

g = Graph()
for i in range(6):
    g.addVertex(i)
    
g.vertList
# print graph
for node in g:
    print(node)

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

# check
for v in g:
   for w in v.getConnections():
       print("( %s , %s )" % (v.getId(), w.getId()))
       
    
#----------------------------------------------------------------
# visualize graph using networkx
#----------------------------------------------------------------
import networkx as nx

g.draw_graph()
g.draw_graph(show_weights=False)
g.draw_graph(DG=False, show_weights=False)

nxDG = g.to_networkx_graph(DG=True)
g.draw_graph(nxDG)
nxG = g.to_networkx_graph(DG=False)
g.draw_graph(nxG)

#----------------------------------------------------------------
nxG.nodes

g.draw_graph(DG=False)
nxG = g.to_networkx_graph(DG=False)
nxG.nodes.data()
nxG.nodes
nxG.edges.data()

list(nxG.neighbors(1))
list(nxG.adj)
nxG.order()




#----------------------------------------------------------------
#  TODO: convert to undirected graph??



#----------------------------------------------------------------
# networkx algorithms
#----------------------------------------------------------------
# shortest path
nx.dijkstra_path(nxG, 5, 0)
nx.dijkstra_path(nxDG, 5, 0)


nxG

#----------------------------------------------------------------
# unvisited nodes
open_set = queue.Queue()
# visited nodes
closed_set = set()
meta = dict()  # key -> (parent state, action to reach child)
start = 1
open_set.put(start)

while open_set.qsize() > 0:
    parent_state = open_set.get()


# check
open_set.queue
open_set.qsize()
open_set.get()
dir(open_set)

  

  
  
#----------------------------------------------------------------
# DEBUG
#----------------------------------------------------------------
from imp import reload

import pygraph
reload(pygraph)
from pygraph import *

g = Graph()
for i in range(6):
    g.addVertex(i)
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

g.draw_graph()
g.draw_graph(show_weights=False)
g.draw_graph(DG=False, show_weights=False)

# g.draw_graph(nxlayout=nx.spring_layout)
# g.draw_graph(nxlayout=nx.random_layout)
# g.draw_graph(nxlayout=nx.circular_layout)
# g.draw_graph(nxlayout=nx.spectral_layout)

g = Graph()
g.addEdge(0,1)
g.addEdge(0,5)
g.addEdge(1,2)
g.addEdge(2,3)

g.draw_graph(DG=False)

pos = nx.shell_layout(G)
nx.draw(G, pos,with_labels=True)


weights = nx.get_edge_attributes(G,'weight')
len(weights)

tmp = g.to_networkx_graph(True)
nx.get_edge_attributes(tmp,'weight')

