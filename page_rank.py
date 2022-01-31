import sys
import numpy as np
import random as rnd
import math
import datetime
import igraph

NUM_OF_VERTICES_G = int(math.pow(2, 10))
NUM_OF_VERTICES_CYCLE = int(math.pow(2, 6))
epsilon = 0.5
N = 2 / epsilon  # int(math.pow(2, 7))  # int(2 / epsilon)  # Path length
t = 2  # default num of iterations
p = 1 / math.pow(2, 6)  # probability for edge creation


def page_rank(G, iterations=t, eps=epsilon):
    n = G.vcount()
    d = np.zeros(n)

    for _ in range(iterations):
        next_vertex = v0 = math.floor(rnd.uniform(0, n))
        for _ in range(N):
            chance_for_rand_neighbor = math.floor(rnd.uniform(0, 1))
            v0_adj_neighbors = G.neighbors(v0, mode="out")
            v0_num_adj_neighbors = len(v0_adj_neighbors)
            if is_visit_random_neighbor(v0, v0_adj_neighbors, v0_num_adj_neighbors, chance_for_rand_neighbor, eps):
                v0_rand_neighbor_indx = math.floor(rnd.uniform(0, v0_num_adj_neighbors))
                next_vertex = v0_adj_neighbors[v0_rand_neighbor_indx]
            else:  # We'll visit a random vertex in G (by Uniform Distribution)
                next_vertex = math.floor(rnd.uniform(0, n))

            v0 = next_vertex
        d[next_vertex] += 1

    return d / iterations


def is_visit_random_neighbor(v0, v0_adj_neighbors, v0_num_adj_neighbors, chance_for_rand_neighbor, eps):
    result = False
    if v0_num_adj_neighbors > 0:
        if (v0_num_adj_neighbors > 1) or (v0 not in v0_adj_neighbors):
            if chance_for_rand_neighbor <= (1 - eps):
                result = True
    return result


def avg_node_degree(G):
    n = G.vcount()
    degrees = [G.degree(i) for i in range(n)]
    return sum(degrees) / n


def incremental_iterations_page_rank(G, eps=epsilon):
    print('\nStarting incremental_iterations_page_rank...')
    time_start = datetime.datetime.now()
    iteration = 1
    t = 2
    d_current = page_rank(G, t, eps)
    print('Completed Page-Rank of t = 2^{0} = {1}'.format(iteration, int(math.pow(2, iteration))))
    print('Finished iteration number: ', iteration)

    while True:
        iteration += 1
        t *= 2
        d_previous = d_current
        d_current = page_rank(G, t)
        dist_len = np.linalg.norm(d_current - d_previous)
        print('Completed Page-Rank of t = 2^{0} = {1}'.format(iteration, int(math.pow(2, iteration))))
        print('||v1 - v0|| = ', dist_len)
        print('Finished iteration number: ', iteration)
        print('---------------------------------------------------------------')
        if dist_len < 1 / math.pow(2, 8):
            break

    total_diff_secs = (datetime.datetime.now() - time_start).total_seconds()
    print('Final iteration is {0}.\nresult: t = 2^{0} = {1}\n'.format(iteration, int(math.pow(2, iteration))))
    print('Final Vector = ', d_current)
    print('Max value in d=', d_current.max())
    print('Indices of max vale: ', *np.where(d_current == d_current.max()))
    print('Min value in d=', d_current.min())
    print('Indices of min vale: ', *np.where(d_current == d_current.min()))
    print('Total Duration: {0} minutes'.format(total_diff_secs / 60))


def convert_nparray_to_igraph(G_nparray):
    G = igraph.Graph(directed=True)
    G.add_vertices(G_nparray.shape[0])
    G.add_edges([(i, j) for i, row in enumerate(G_nparray) for j, _ in enumerate(row) if G_nparray[i, j] > 0])
    return G


def create_random_directed_graph():
    return convert_nparray_to_igraph(create_random_numpyarr())


def create_random_numpyarr():
    G_array = np.random.randint(low=0, high=2, size=(NUM_OF_VERTICES_G, NUM_OF_VERTICES_G))
    return G_array


def create_random_igraph_with_probability(is_dedicated_probability=False):
    prob = p
    G_array = np.zeros((NUM_OF_VERTICES_G, NUM_OF_VERTICES_G))
    for i, row in enumerate(G_array):
        for j, _ in enumerate(row):
            if is_dedicated_probability:
                prob = 1 / math.log(j + 2, 2)
            if rnd.random() <= prob:
                G_array[i, j] = 1.0
    return convert_nparray_to_igraph(G_array)


def create_cycle_graph_and_add_edge(G):
    n_of_G = G.vcount()
    n_of_C = NUM_OF_VERTICES_CYCLE
    C = igraph.Graph.Ring(n_of_C, directed=True)
    G.add_vertices(n_of_C)
    G.add_edges([(k + n_of_G, l + n_of_G) for (k, l) in C.get_edgelist()])
    u_of_edge = np.random.randint(low=0, high=n_of_G)
    v_of_edge = np.random.randint(low=n_of_G, high=n_of_G + n_of_C)
    new_edges = [(u_of_edge, v_of_edge)]
    G.add_edges(new_edges)
    return G


def incremental_epsilon_test(G):
    eps = epsilon
    print('Running incremental Epsilon page_ranks...')
    while eps >= 1 / math.pow(2, 10):
        print('Current epsilon=1/2^{0}'.format(-math.log(eps, 2)))
        incremental_iterations_page_rank(G, eps)
        print('Finished iteration for epsilon=1/2^{0}'.format(-math.log(eps, 2)))
        eps *= 0.5
        print('********************************************')

    # TODO: Need to clear up garbage code from 'main'
if __name__ == '__main__':
        NUM_OF_VERTICES_G = int(math.pow(2, 10))
        NUM_OF_VERTICES_CYCLE = int(math.pow(2, 6))
        N = int(math.pow(2, 7))  # int(2 / epsilon)  # Path length
        t = 2  # default num of iterations
        p = 1 / math.pow(2, 6)  # probability for edge creation
        print('Parameters for current test:')
        print('p=1/2^{0}, epsilon={1}, N=2^{2}\n'.format(-math.log(p, 2), epsilon, (math.log(N, 2))))
        print('Parameters for Graph creation:')
        print('NUM_OF_VERTICES_G=2^{0}, NUM_OF_VERTICES_CYCLE=2^{1}\n'.format(math.log(NUM_OF_VERTICES_G, 2),
                                                                              math.log(NUM_OF_VERTICES_CYCLE, 2)))
        G = create_random_igraph_with_probability()
        G = create_cycle_graph_and_add_edge(G)
        print('Average degree: ', avg_node_degree(G))
        # d = page_rank(G)
        # incremental_iterations_page_rank(G)
        incremental_epsilon_test(G)
        print('------------------------------------')