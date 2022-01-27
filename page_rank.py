import numpy as np
import networkx as nx
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
# DEBUG:
time0 = datetime.datetime.now()
time1 = datetime.datetime.now()


def page_rank(G, iterations=t):
    n = G.vcount()
    d = [0 for i in range(n)]

    for _ in range(iterations):
        # v0 = math.floor(np.random.uniform(low=0, high=n))
        v0 = math.floor(rnd.uniform(0, n))
        for _ in range(N):
            # chance_for_rand_neighbor = math.floor(np.random.uniform(low=0, high=1))
            chance_for_rand_neighbor = math.floor(rnd.uniform(0, 1))
            v0_num_neighbors = len(G.neighbors(v0))
            if v0_num_neighbors > 0 and chance_for_rand_neighbor <= (1 - epsilon):  # Then we'll visit to a random neighbor
                # v0_rand_neighbor_indx = math.floor(np.random.uniform(low=0, high=v0_num_neighbors))
                v0_rand_neighbor_indx = math.floor(rnd.uniform(0, v0_num_neighbors))
                next_vertex = G.neighbors(v0)[v0_rand_neighbor_indx]
            else:  # We'll visit a random vertex in G (by Uniform Distribution)
                # next_vertex = math.floor(np.random.uniform(low=0, high=n))
                next_vertex = math.floor(rnd.uniform(0, n))

            v0 = next_vertex
        d[next_vertex] += 1
    d = [k / iterations for k in d]

    return d


def avg_node_degree(G):
    arr = [0 for i in range(NUM_OF_VERTICES)]
    # arr = np.zeros(NUM_OF_VERTICES)
    sum_degrees = 0
    for v in range(NUM_OF_VERTICES):
        arr[v] = len(G.neighbors(v))
        sum_degrees += arr[v]
    print(arr)
    return sum_degrees / NUM_OF_VERTICES


# TODO - [Performance:] moving from networkx igraph improved by 4 times faster - but still not good enough!!!
def incremental_iterations_page_rank(G):
    time_start = datetime.datetime.now()
    iteration = 1
    # DEBUG:
    t = 2
    # start_timer()
    #
    # d0 = np.array(page_rank(G, t))
    # # DEBUG:
    # diff = get_time_passed_seconds()
    # print("Page-Rank calc of t = {1} took: {0} seconds.".format(diff, t))
    #
    # t *= 2
    # iteration += 1
    # # DEBUG:
    start_timer()

    d_current = np.array(page_rank(G, t))
    # DEBUG:
    diff = get_time_passed_seconds()
    print("Page-Rank calc of t = {1} took: {0} seconds.".format(diff, t))

    t *= 2
    iteration += 1

    while True:
        d_previous = d_current

        # DEBUG:
        start_timer()

        d_current = np.array(page_rank(G, t))
        # DEBUG:
        diff = get_time_passed_seconds()
        print("Page-Rank calc took: {0} seconds.".format(diff))

        # DEBUG:
        start_timer()

        dist_len = np.linalg.norm(d_current - d_previous)
        # DEBUG:
        diff = get_time_passed_seconds()
        print("Distance calc took: {0} seconds.".format(diff))

        print('Done page rank for t = 2^{0} = {1}'.format(iteration, math.floor(math.pow(2, iteration))))
        print('||v1 - v0|| = ', dist_len)
        print('Finished iteration number: ', iteration)
        print('---------------------------------------------------------------')

        if dist_len < 1 / math.pow(2, 8):
            break
        iteration += 1
        t *= 2

    total_diff = (datetime.datetime.now() - time_start).total_seconds()
    print('Final iteration was: {0} of t = 2^{0} = {1}'.format(iteration, math.floor(math.pow(2, iteration))))
    print('Total Duration: {0} minutes'.format(total_diff / 60))


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
    connecting_edge = tuple(u_of_edge, v_of_edge)
    G.add_edges(list(connecting_edge))
    return G


# DEBUG:
def start_timer():
    time0 = datetime.datetime.now()


# DEBUG:
def get_time_passed_seconds():
    time1 = datetime.datetime.now()
    diff = time1 - time0
    start_timer()
    # return divmod(diff.total_seconds(), 60)
    return diff.total_seconds()


# TODO: Need to clear up garbage code from 'main'
if __name__ == '__main__':
    # for i in range(10):
    G = create_random_igraph_with_probability()
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
