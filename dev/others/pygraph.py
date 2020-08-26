import networkx as nx
import matplotlib.pyplot as plt
import sys
import queue
import heapq


class Vertex(object):
    "node class"
    def __init__(self, key):
        # vertex id
        self.key = key
        # neighouring vertices
        self.connectedTo = {}
        self.data = {}

    def addNeighbor(self, nbr, weight=0):
        "connect another vertex object"
        self.connectedTo[nbr] = weight

#     def __str__(self):
#         "string representation"
#         return str(self.key) + ' connectedTo: ' + str([x.key for x in self.connectedTo])

    def getConnections(self):
        "find all neighbours"
        return list(self.connectedTo.keys())


    def getWeight(self, nbr):
        "get weight of the connecting edge"
        return self.connectedTo[nbr]

    def set_data(self, key, value):
        "store value"
        self.data[key] = value

    def get_data(self, key):
        "return data value"
        return (self.data[key])

    def clear_all_data(self):
        "clear data dict"
        self.data = {}

    def get_all_data(self):
        "print data dict"
        return self.data

    @property
    def id(self):
        return self.key

    @id.setter
    def id(self, value):
        self.key = value

    def __hash__(self):
        "make the class hashable"
        return hash(self.id)

    def __lt__(self, other):
        "for vertex comparison"
        return (self.data['distance'] < other.data['distance'])

    def __eq__(self, other):
        "for vertex comparison"
        return (self.data['distance'] == other.data['distance'])

class DiGraph(object):
    "digraph"
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        "add vertec object with key as ID"
        self.numVertices = self.numVertices + 1

        newVertex = Vertex(key)
        # FIXME: duplicated vertex??
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        "get vertex identified by ID"
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        "check if vertex n exists"
        return n in self.vertList

    def addEdge(self, f, t, cost=1):
        "add edge connecting the two vertices"
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        # use vertex object to create edges
        self.vertList[f].addNeighbor(self.vertList[t], cost)

    def getVerticesID(self):
        "get all vertex IDs"
        return list(self.vertList)

    def getVertices(self):
        "get dictionary of vertices"
        return self.vertList

    def getEdges(self, show_weight=False):
        "get list of connected edges"
        if show_weight == False:
            return ([(node.id, neighbour.id) for node in self.vertList.values()
                 for neighbour in node.connectedTo])
        else:
            return ([(node.id, neighbour.id, node.getWeight(neighbour))
                     for node in self.vertList.values()
                     for neighbour in node.connectedTo])

    def __iter__(self):
        "print vertex values"
        return iter(self.vertList.values())

    def get_all_data(self):
        "get data of all vertices"
        return [(v.id, v.get_all_data())
                for v in self.getVertices().values()]

    def clear_all_data(self):
        "clear all data in all vertices"
        for v in self.getVertices().values():
            v.clear_all_data()

    def shortest_path(self, nodeID, targetID):
        "shortest path algorithm"

        # assign distance for each vertex in graph
        unvisitedQueue = queue.PriorityQueue()
        parent = {}
        visitedNodeID = set()
        vertexDict  = self.getVertices()

        # initialize distance
        for id, V in vertexDict.items():
            if id == nodeID:
                V.set_data('distance', 0)
            else:
                V.set_data('distance', sys.maxsize)
            parent[id] = None
            unvisitedQueue.put(V)

        while unvisitedQueue.qsize() > 0:
            V = unvisitedQueue.get()
            visitedNodeID.add(V.id)

            for nbr in self.neighbours(V.id):
                # calculate distance
                dist = V.get_data('distance') + V.getWeight(nbr)
                # if this distance is shorter
                if dist < nbr.get_data('distance'):
                    nbr.set_data('distance', dist)
                    parent[nbr.id] = V.id

            # clear the queue
            while not unvisitedQueue.empty():
                unvisitedQueue.get()

            # build the queue again
            unvisitedNode = [v for id, v in vertexDict.items() if id not in visitedNodeID]
            for V in unvisitedNode:
                unvisitedQueue.put(V)

        # print the shortest path now
        currentNodeID = targetID
        path = []
        totalWeights = 0

        while parent[currentNodeID] is not None:
            path.append(currentNodeID)
            # add the weight
            parentNode = vertexDict[parent[currentNodeID]]
            currentNode = vertexDict[currentNodeID]
            totalWeights += parentNode.getWeight(currentNode)
            currentNodeID = parent[currentNodeID]

        path.append(currentNodeID)
        path.reverse()

        return (path, totalWeights)



    def find_isolated_nodes(self):
        "get list of isolated nodes"
        vertexDict = self.getVertices()
        isolatedNodes = []
        for id, node in vertexDict.items():
            if len(node.getConnections()) == 0:
                isolatedNodes.append(node)

        return (isolatedNodes)

    def order(self):
        "return number of nodes"
        return (len(self.getVertices()))

    def size(self, weighted=True):
        "return num of edges and total weights"
        edges = self.getEdges(True)
        totalWeights = sum([item[2] for item in edges])
        if weighted:
            return totalWeights
        else:
            return len(edges)

    def density(self):
        "2E / (V*(V-1))"
        numEdges = self.size(False)
        numVertices = self.order()
        return (2.0 * numEdges / (numVertices * (numVertices-1)))

    def min_degree(self):
        "minimum degree of graph"
        pass

    def max_degree(self):
        "maximum degree of graph"
        pass

    def neighbours(self, nodeID):
        "list neighbours"

        neighbour_nodes = self.vertList[nodeID].getConnections()
        return (neighbour_nodes)

    def diameter(self):
        "diameter of the graph"
        # diameter is the maximum length from one to node to another node
        # find the shortest path for each pair of nodes
        vertexDict = self.getVertices()
        diameter = None
        for srcID, v1 in vertexDict.items():
            for targetID, v2 in vertexDict.items():
                if targetID != srcID:
                    (path, weights) = self.shortest_path(srcID, targetID)
                    if diameter is None:
                        diameter = weights
                    if diameter < weights:
                        diameter = weights


        return diameter


    def is_connected(self):
        "determines if the graph is a connected graph"
        # a graph is connected if every pair of nodes are connected
        vertexDict = self.getVertices()
        numVertices = len(vertexDict)
        T = Task(self, actionList=[])

        # do a bfs for each vertex
        for id, v in vertexDict.items():
            metaData = bfs(T, id)
            if (len(metaData) < numVertices):
                return False

        return True

    def to_networkx_graph(self, DG=True):
        "convert to networkx"

        if DG == True:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        # assumed weights
        G.add_weighted_edges_from(self.getEdges(True))
        return G

    def draw_graph(self, G=None, DG=True, show_weights=True, nxlayout=nx.shell_layout):
        "draw network graph"

        if G is None:
            G = self.to_networkx_graph(DG)

        # get weights
        weights = nx.get_edge_attributes(G,'weight')
        # define layout
        pos = nxlayout(G)

        nx.draw(G, pos, with_labels=True)

        if show_weights == True:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)


class Task(object):
    "Task to do for each node in G"
    def __init__(self, G, actionList=None, targetNode=None):
        self.G = G
        # list of actions to perform with the graph
        self.actionList = actionList
        self.targetNode = targetNode

    def is_goal(self, nodeID):
        "check if the target node is found"
        if self.targetNode is not None:
            return (nodeID == self.tagetNode)
        else:
            return False

def bfs(T, nodeID):
    "breadth first search"

    G = T.G
    # a FIFO open_set
    open_set = queue.Queue()
    # an empty set to maintain visited nodes
    closed_set = set()
    # a dictionary to maintain meta information
    # key -> (parent state, info
    meta = dict()
    open_set.put(nodeID)
    counter = 0
    meta[nodeID] = (None, counter)

    while open_set.qsize() > 0:

        current_nodeID = open_set.get()

        V = G.getVertex(current_nodeID)
        for action in T.actionList:
            action(V)

        # termination node
        if T.is_goal(current_nodeID):
            return meta


        # find children nodes
        for child_node in G.neighbours(current_nodeID):
          # skip if already visited
          if child_node.id in closed_set:
              continue

          # if not visited, add it to the queue
          if child_node.id not in open_set.queue:
              # store the parent
              counter += 1
              # store the order of discovered nodes
              meta[child_node.id] = (current_nodeID, counter)
              open_set.put(child_node.id)

        closed_set.add(current_nodeID)

    return meta


def parse_path(metaData):
    "find the traversed path"
    pathList = []
    for key in metaData.keys():
        pathList.append((key, metaData[key][1]))

    # sort by parse order
    pathList = sorted(pathList, key=lambda x: x[1])
    return pathList


def dfs(T, nodeID):
    "depth first search"

    def traverse_till_end(T, vertex, parentID, closed_set, meta, counter):
        "traverse to the end"

        # skip if already visited
        if vertex.id in closed_set:
            return (T, closed_set, meta, counter)

        for action in T.actionList:
            action(vertex)

        # store the order of discovered nodes
        counter += 1
        meta[vertex.id] = (parentID, counter)
        closed_set.add(vertex.id)

        for child_node in T.G.neighbours(vertex.id):
              (T, closed_set, meta, counter) = traverse_till_end(T, child_node, vertex.id, closed_set, meta, counter)

        return (T, closed_set, meta, counter)

    # an empty set to maintain visited nodes
    closed_set = set()
    # a dictionary to maintain meta information
    # key -> (parent state, info
    meta = dict()
    # starting node
    #open_set.put(nodeID)
    counter = 0
    # record path
    meta[nodeID] = (None, counter)

    #current_nodeID = open_set.get()
    V = T.G.getVertex(nodeID)
    for action in T.actionList:
        action(V)

    closed_set.add(nodeID)

    # recursive call
    for child_node in T.G.neighbours(nodeID):
        (T, closed_set, meta, counter) = traverse_till_end(T, child_node, nodeID, closed_set, meta, counter)

    return meta


def shortest_path(G, nodeID, targetID):
    "find the shortest path to all nodes starting from nodeID"

    # assign distance for each vertex in graph
    unvisitedQueue = queue.PriorityQueue()
    parent = {}
    visitedNodeID = set()

    vertexDict  = G.getVertices()

    # initialize distance
    for id, V in vertexDict.items():
        if id == nodeID:
            V.set_data('distance', 0)
        else:
            V.set_data('distance', sys.maxsize)
        parent[id] = None
        unvisitedQueue.put(V)


    while unvisitedQueue.qsize() > 0:
        V = unvisitedQueue.get()
        visitedNodeID.add(V.id)

        for nbr in G.neighbours(V.id):
            # calculate distance
            dist = V.get_data('distance') + V.getWeight(nbr)
            # if this distance is shorter
            if dist < nbr.get_data('distance'):
                nbr.set_data('distance', dist)
                parent[nbr.id] = V.id

        # clear the queue
        while not unvisitedQueue.empty():
            unvisitedQueue.get()

        # build the queue again
        unvisitedNode = [v for id, v in vertexDict.items() if id not in visitedNodeID]
        for V in unvisitedNode:
            unvisitedQueue.put(V)


    # print the shortest path now
    currentNodeID = targetID
    path = []
    totalWeights = 0

    while parent[currentNodeID] is not None:
        path.append(currentNodeID)
        # add the weight
        parentNode = vertexDict[parent[currentNodeID]]
        currentNode = vertexDict[currentNodeID]
        totalWeights += parentNode.getWeight(currentNode)
        currentNodeID = parent[currentNodeID]

    path.append(currentNodeID)
    path.reverse()

    return (parent, path, totalWeights)
