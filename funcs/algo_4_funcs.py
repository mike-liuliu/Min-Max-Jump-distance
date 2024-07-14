from sklearn.metrics import pairwise_distances
import numpy as np
import networkx as nx


def construct_MST_from_graph(distance_matrix):
    
    lenX = len(distance_matrix)
    g = construct_MST_prim(lenX)
    g.graph = distance_matrix
    MST_list = g.primMST()    
 
    MST = nx.Graph()
    for i in range(lenX):
        MST.add_node(i)
    for edge in MST_list:
        MST.add_edge(edge[0],edge[1],weight=edge[2])          
    return MST


def cal_mmj_matrix_by_algo_4_Calculation_and_Copy(X, round_n = 15):
    
    
    lenX = len(X)
    distance_matrix = pairwise_distances(X)
    distance_matrix = np.round(distance_matrix,round_n)
    mmj_matrix = np.zeros((lenX,lenX))

    MST = construct_MST_from_graph(distance_matrix)
    
    MST_edge_list = list(MST.edges(data='weight'))
 
    edge_node_list = [(edge[0],edge[1]) for edge in MST_edge_list]
    edge_weight_list = [edge[2] for edge in MST_edge_list]
    edge_large_to_small_arg = np.argsort(edge_weight_list)[::-1]
    edge_weight_large_to_small = np.sort(edge_weight_list)[::-1]
    edge_nodes_large_to_small = [edge_node_list[i] for i in edge_large_to_small_arg]
 
    for i, edge_nodes in enumerate(edge_nodes_large_to_small):
        edge_weight = edge_weight_large_to_small[i]
        MST.remove_edge(*edge_nodes)
        tree1_nodes = list(nx.dfs_preorder_nodes(MST, source=edge_nodes[0]))
        tree2_nodes = list(nx.dfs_preorder_nodes(MST, source=edge_nodes[1]))
        for p1 in tree1_nodes:
            for p2 in tree2_nodes:
                mmj_matrix[p1, p2] = mmj_matrix[p2, p1] = edge_weight      
 
    return mmj_matrix


# Prim's Minimum Spanning Tree (MST) algorithm. 
# Based on the code from geeksforgeeks.org. See:
# https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/

 
import sys
class construct_MST_prim():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = None

    # A utility function to print 
    # the constructed MST stored in parent[]
    def printMST(self, parent):
#         print("Edge \tWeight")
        
        MST = []
        for i in range(1, self.V):
#             print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
            MST.append([parent[i],i, self.graph[i][parent[i]]])
        return MST

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):

        # Initialize min value
        min = sys.maxsize

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):

        # Key values used to pick minimum weight edge in cut
        key = [sys.maxsize] * self.V
        parent = [None] * self.V # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1 # First node is always the root of

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):

                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > 0 and mstSet[v] == False \
                and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        MST_list = self.printMST(parent)
        
        return MST_list