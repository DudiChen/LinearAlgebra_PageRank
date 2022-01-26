import numpy as np
import networkx as nx
import random as rnd
import math

NUM_OF_VERTICES = int(math.pow(2, 10))
NUM_OF_VERTICES_CYCLE = 6  # int(math.pow(2, 6))
epsilon = 0
N = int(math.pow(2, 6))  # int(2 / epsilon)  # Path length
t = 2  # Num of iterations
p = 1 / math.pow(2, 6)


def page_rank(G, iterations=t):
    n = G.number_of_nodes()
    d = [0 for i in range(n)]
    v0 = math.floor(np.random.uniform(low=0, high=n))
    for _ in range(iterations):
        for _ in range(N):
            chance_for_rand_neighbor = rnd.random()
            v0_num_neighbors = len(G[v0])
            if chance_for_rand_neighbor <= (
                    1 - epsilon) and v0_num_neighbors > 0:  # Then we'll visit to a random neighbor
                v0_rand_neighbor_indx = math.floor(np.random.uniform(low=0, high=v0_num_neighbors))
                next_vertex = list(G[v0].keys())[v0_rand_neighbor_indx]
            else:  # We'll visit a random vertex in G (by Uniform Distribution)
                next_vertex = math.floor(np.random.uniform(low=0, high=n))

            v0 = next_vertex
        d[next_vertex] += 1
    d = [k / t for k in d]

    return d


def avg_node_degree(G):
    arr = [0 for i in range(NUM_OF_VERTICES)]
    sum_degrees = 0
    print(arr)
    for v in range(NUM_OF_VERTICES):
        arr[v] = len(G[v])
        sum_degrees += len(G[v])
    print(arr)
    return sum_degrees / NUM_OF_VERTICES

#TODO - find where's the bug
def incremental_iterations_page_rank(G):
    t = 2
    d0 = page_rank(G, t)
    t *= 2
    d1 = page_rank(G, t)
    t *= 2
    loop = 1

    while np.linalg.norm(np.array(d1) - np.array(d0)) >= 1 / math.pow(2, 8):
        d0 = d1
        d1 = page_rank(G, t)
        print('Done page rank for t = ', t)
        print('||d1 - d0|| = ', np.linalg.norm(np.array(d1) - np.array(d0)))
        print('done loop number', loop)
        print('---------------------------------------------------------------')
        loop += 1
        t *= 2
    print('Final t is:', t)


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
                G_array[i, j] = 1.0
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
    # for i in range(10):
    G = create_random_DiGraph_with_probability()
    print(avg_node_degree(G))
    incremental_iterations_page_rank(G)

    # d = page_rank(G, 8)
    # print(d)
    # print("sum of d:")
    # print(sum(d))
    print('------------------------------------')

    # G = create_cycle_graph_and_add_edge(G)
    # print('After:')
    # print(G.nodes)
    # print(G.edges)

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
