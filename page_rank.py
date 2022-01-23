import numpy as np
import networkx as nx
import random as rnd

epsilon = .64
N = 5
t = 100


def page_rank(G):
    n = G.shape[0]
    d = [0 for i in range(n)]
    v0 = np.random.uniform(low=0, high=n)

    for i in range(N):
        chance_for_rand_neighbor = rnd.random()
        if chance_for_rand_neighbor <= (1 - epsilon): # Then we'll visit to a random neighbor
            pass
        else: # We'll visit a random vertex in G (by Uniform Distribution)
            pass





if __name__=='__main__':
    # Example extracted from 'Introduction to Information Retrieval'
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
    # iter = G.neighbors(1)
    print(len([i for i in G[3]]))
    print(nx.to_numpy_array(G))
    G.add_edge(1,0)
    print('---------------------------')
    print(nx.to_numpy_array(G))

    # G = np.array([[1,0,1,0,0,0,0],
    #               [0,1,1,0,0,1,0],
    #               [1,0,1,0,0,0,0],
    #               [0,0,0,1,1,0,0],
    #               [0,1,0,0,0,0,1],
    #               [0,0,1,0,0,1,1],
    #               [0,0,0,1,1,0,1]])
   # print(page_rank(G))



#   print([i for i in G.neighbors(3)])