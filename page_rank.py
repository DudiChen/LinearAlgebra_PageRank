import numpy as np
import networkx as nx
import random as rnd
import math

NUM_OF_VERTICES = 10  # int(math.pow(2, 10))
NUM_OF_VERTICES_CYCLE = 6  # int(math.pow(2, 6))
epsilon = .64
N = 5  # Path length
t = 100  # Num of iterations
p = 1 / math.pow(2, 6)


def page_rank(G):
    n = G.number_of_nodes()
    d = [0 for i in range(n)]
    v0 = math.floor(np.random.uniform(low=0, high=n))
    for _ in range(t):
        for _ in range(N):
            chance_for_rand_neighbor = rnd.random()
            if chance_for_rand_neighbor <= (1 - epsilon):  # Then we'll visit to a random neighbor
                v0_num_neighbors = len(G[v0])
                v0_rand_neighbor_indx = math.floor(np.random.uniform(low=0, high=v0_num_neighbors))
                next_vertex = list(G[v0].keys())[v0_rand_neighbor_indx]
            else:  # We'll visit a random vertex in G (by Uniform Distribution)
                next_vertex = math.floor(np.random.uniform(low=0, high=n))

            v0 = next_vertex
        d[next_vertex] += 1
    d = [k / t for k in d]

    return d


def create_random_DiGraph():
    G_array = np.random.randint(low=0, high=2, size=(NUM_OF_VERTICES, NUM_OF_VERTICES))
    return nx.DiGraph(G_array)


def create_random_numpyarr():
    G_array = np.random.randint(low=0, high=2, size=(NUM_OF_VERTICES, NUM_OF_VERTICES))
    return G_array


def create_random_DiGraph_with_probability(is_dedicated_probability=False):
    prob = p
    G_array = np.zeros((NUM_OF_VERTICES, NUM_OF_VERTICES))
    for i, row in enumerate(G_array):
        for j, _ in enumerate(row):
            if is_dedicated_probability:
                prob = 1 / math.log(j + 2, 2)
            if rnd.random() <= prob:
                G_array[i, j] = 1
    return nx.DiGraph(G_array)


def create_cycle_graph_and_add_edge(G):
    C = nx.cycle_graph(n=NUM_OF_VERTICES_CYCLE, create_using=nx.DiGraph)
    G.add_nodes_from([k + NUM_OF_VERTICES for k in list(C.nodes)])
    G.add_edges_from([(k + NUM_OF_VERTICES, l + NUM_OF_VERTICES) for (k, l) in list(C.edges)])
    G.add_edge(u_of_edge=np.random.randint(low=0, high=NUM_OF_VERTICES),
               v_of_edge=np.random.randint(low=NUM_OF_VERTICES, high=NUM_OF_VERTICES + NUM_OF_VERTICES_CYCLE))
    return G


# TODO: Need to add a function to create a randomized graph with selfloops
# TODO: Need to add page_rank executions as per assignment questions
# TODO: Need to clear up garbage code from 'main'
if __name__ == '__main__':
    for i in range(20):
        G = create_random_DiGraph_with_probability(True)
        G = create_cycle_graph_and_add_edge(G)
        print('After:')
        print(G.nodes)
        print(G.edges)
        print('------------------------------------')




    # print('before:')
    # print(G.nodes)
    # print(G.edges)
    # print([k + NUM_OF_VERTICES for k in list(C.nodes)])
    # print([(k + NUM_OF_VERTICES, l + NUM_OF_VERTICES) for (k, l) in list(C.edges)])

    # print('G')
    # print(nx.to_numpy_array(G))
    # print('C')
    # print(nx.to_numpy_array(C))
    # F = nx.compose(G, C)
    # print('F')
    # print(nx.to_numpy_array(F))

    # for i in range(10):
    #     # G = create_random_numpyarr()
    #     # print(len(G))
    #     print(list(G[5].keys())[0])
    #     # print(G_List[0])
    #     # print(G_List[3])
    #     # print(G_List[20])
    #     # print(G_List[151])
    #
    #     # print(nx.to_numpy_array(G))
    #     # print(nx.to_numpy_array(G))
    #     print("Page Rank for G:")
    #     # d = page_rank(G)
    #     # print(len(d))
    #     # print("sum of d:")
    #     # print(sum(d))
    #     # print('---------------------------')

    # G = nx.DiGraph(
    #     np.array([
    #         [1, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 1, 0],
    #         [1, 0, 1, 0, 0, 0, 0],
    #         [1, 0, 1, 0, 1, 1, 1],
    #         [0, 1, 0, 0, 0, 0, 1],
    #         [0, 0, 1, 0, 0, 1, 1],
    #         [0, 0, 0, 1, 1, 0, 1]
    #     ])
    # )
