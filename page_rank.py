import numpy as np
import random as rnd
import math
import datetime
import igraph

NUM_OF_VERTICES = int(math.pow(2, 10))
NUM_OF_VERTICES_CYCLE = 6  # int(math.pow(2, 6))
epsilon = 0
N = int(math.pow(2, 6))  # int(2 / epsilon)  # Path length
t = 2  # Num of iterations
p = 1 / math.pow(2, 6)


def page_rank(G, iterations=t):
    n = G.vcount()
    d = np.zeros(n)

    for _ in range(iterations):
        v0 = math.floor(rnd.uniform(0, n))
        for _ in range(N):
            chance_for_rand_neighbor = math.floor(rnd.uniform(0, 1))
            v0_num_neighbors = len(G.neighbors(v0))
            if v0_num_neighbors > 0 and chance_for_rand_neighbor <= (1 - epsilon):  # Then we'll visit to a random neighbor
                v0_rand_neighbor_indx = math.floor(rnd.uniform(0, v0_num_neighbors))
                next_vertex = G.neighbors(v0)[v0_rand_neighbor_indx]
            else:  # We'll visit a random vertex in G (by Uniform Distribution)
                next_vertex = math.floor(rnd.uniform(0, n))

            v0 = next_vertex
        d[next_vertex] += 1

    return d / iterations


def avg_node_degree(G):
    degrees = [len(G.neighbors(i)) for i in range(NUM_OF_VERTICES)]
    return sum(degrees) / NUM_OF_VERTICES


def incremental_iterations_page_rank(G):
    time_start = datetime.datetime.now()
    iteration = 1
    t = 2
    d_current = page_rank(G, t)
    t *= 2
    iteration += 1
    while True:
        d_previous = d_current
        d_current = page_rank(G, t)
        dist_len = np.linalg.norm(d_current - d_previous)
        print('Done page rank for t = 2^{0} = {1}'.format(iteration, int(math.pow(2, iteration))))
        print('||v1 - v0|| = ', dist_len)
        print('Finished iteration number: ', iteration)
        print('---------------------------------------------------------------')
        if dist_len < 1 / math.pow(2, 8):
            break
        iteration += 1
        t *= 2

    total_diff_secs = (datetime.datetime.now() - time_start).total_seconds()
    print('Final iteration was: {0} of t = 2^{0} = {1}'.format(iteration, int(math.pow(2, iteration))))
    print('Total Duration: {0} minutes'.format(total_diff_secs / 60))


def convert_nparray_to_igraph(G_nparray):
    G = igraph.Graph(directed=True)
    G.add_vertices(G_nparray.shape[0])
    G.add_edges([(i, j) for i, row in enumerate(G_nparray) for j, _ in enumerate(row) if G_nparray[i, j] > 0])
    return G


def create_random_directed_graph():
    return convert_nparray_to_igraph(create_random_numpyarr())


def create_random_numpyarr():
    G_array = np.random.randint(low=0, high=2, size=(NUM_OF_VERTICES, NUM_OF_VERTICES))
    return G_array


def create_random_igraph_with_probability(is_dedicated_probability=False):
    prob = p
    G_array = np.zeros((NUM_OF_VERTICES, NUM_OF_VERTICES))
    for i, row in enumerate(G_array):
        for j, _ in enumerate(row):
            if is_dedicated_probability:
                prob = 1 / math.log(j + 2, 2)
            if rnd.random() <= prob:
                G_array[i, j] = 1.0
    return convert_nparray_to_igraph(G_array)


def create_cycle_graph_and_add_edge(G):
    C = igraph.Graph.Ring(NUM_OF_VERTICES_CYCLE, directed=True)
    G.add_vertices(NUM_OF_VERTICES_CYCLE)
    G.add_edges([(k + NUM_OF_VERTICES, l + NUM_OF_VERTICES) for (k, l) in C.get_edgelist()])
    u_of_edge = np.random.randint(low=0, high=NUM_OF_VERTICES)
    v_of_edge = np.random.randint(low=NUM_OF_VERTICES, high=NUM_OF_VERTICES + NUM_OF_VERTICES_CYCLE)
    new_edges = [(u_of_edge, v_of_edge)]
    G.add_edges(new_edges)
    return G


# TODO: Need to clear up garbage code from 'main'
if __name__ == '__main__':
    G = create_random_igraph_with_probability()
    print(avg_node_degree(G))
    incremental_iterations_page_rank(G)
    print('------------------------------------')


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
