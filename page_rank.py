import numpy as np
import networkx as nx
import random as rnd
import math

epsilon = .64
N = 5
t = 100


def page_rank(G):
    n = G.number_of_nodes()
    d = [0 for i in range(n)]
    v0 = math.floor(np.random.uniform(low=0, high=n))
    for _ in range(t):
        for _ in range(N):
            chance_for_rand_neighbor = rnd.random()
            if chance_for_rand_neighbor <= (1 - epsilon):   # Then we'll visit to a random neighbor
                v0_num_neighbors = len(G[v0])
                v0_rand_neighbor_indx = math.floor(np.random.uniform(low=0, high=v0_num_neighbors))
                next_vertex = list(G[v0].keys())[v0_rand_neighbor_indx]
            else:   # We'll visit a random vertex in G (by Uniform Distribution)
                next_vertex = math.floor(np.random.uniform(low=0, high=n))

            v0 = next_vertex
        d[next_vertex] += 1
    d = [k/t for k in d]

    return d


# TODO: Need to add a function to create a randomized graph with selfloops
# TODO: Need to add page_rank executions as per assignment questions
# TODO: Need to clear up garbage code from 'main'
if __name__ == '__main__':
    G = nx.DiGraph(
        np.array([
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 1]
        ])
    )
    print(nx.to_numpy_array(G))
    G.add_edge(1, 0)
    print('---------------------------')
    print(nx.to_numpy_array(G))
    print("Page Rank for G:")
    d = page_rank(G)
    print(d)
    print("sum of d:")
    print(sum(d))

