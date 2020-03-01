import numpy as np
import tsplib95 as tsp
import tspdraw
from pathlib import Path
from numba import jit, njit, prange
from random import random
from time import time
from parse_solutions import get_solutions


#def distance_matrix(coords):
#    n = max(coords.keys()) + 1
#    mat = np.zeros((n,n))
#    for i in coords:
#        coords[i] = np.array(coords[i])
#    for i, a in coords.items():
#        for j, b in coords.items():
#            mat[i,j] = np.linalg.norm(a - b)
#    return mat


def distance_matrix(*args):
    #n = max(coords.keys()) + 1
    #mat = np.zeros((n,n))
    mat = {}
    for i, a in enumerate(problem.get_nodes()):
        for j, b in enumerate(problem.get_nodes()):
            #mat[i+1,j+1] = problem.wfunc(a,b)
            mat[a,b] = problem.wfunc(a,b)
    return mat


def pairwise(tour):
    tour = iter(tour)
    one = next(tour)
    first = one
    try:
        while two := next(tour):
            yield one, two
            one = two
    except StopIteration:
        yield one, first  # complete cycle


@njit
def validate_individual(i):
    assert i[0] == 1
    assert np.unique(i).size == i.size


@njit
def validate_pop(p):
    for i in range(len(p)):
        validate_individual(p[i])


@njit
def init_population(n, nodes):
    p = np.empty((n,nodes.size), dtype=nodes.dtype)
    p[:] = nodes
    for i in range(p.shape[0]):
        np.random.shuffle(p[i,1:])
    validate_pop(p)
    return p


@njit
def mutate(a):
    i = int(random() * a.size - 1) + 1
    j = int(random() * a.size - 1) + 1
    #if i < j:
    #    i,j = j,i
    #np.random.shuffle(a[i:j])
    tmp = a[i]
    a[i] = a[j]
    a[j] = tmp


@njit
def mutate_all(pop, prob):
    for i in range(pop.shape[0]):
        if random() < prob:
            mutate(pop[i])


def dist_vec(population):
    vec = np.empty(len(population))
    for i,tour in enumerate(population):
        #vec[i] = sum(problem.wfunc(a,b) for a,b in pairwise(tour))
        #vec[i] = sum(graph[a][b]['weight'] for a,b in pairwise(tour))
        vec[i] = sum(dmat[a,b] for a,b in pairwise(tour))
    return vec


@njit
def dist_to_fitness(vec):
    pdist = vec / vec.max()
    pdist = 1 - pdist
    pdist **= 2
    pdist /= pdist.sum()
    return pdist


@njit
def splice(a, b, out=None):
    if out is None:
        out = np.empty_like(a)
    n = a.size
    i = int(random() * (n-1))
    j = int(random() * (n-1))
    if i == j: j = (j+1) % n
    if i > j: i, j = j, i
    out[i:j] = a[i:j]
    s = set(a[i:j])
    k = j
    while k != i:
        while b[j] in s:
            j = (j+1) % n
        out[k] = b[j]
        j = (j+1) % n
        k = (k+1) % n
    val = np.argwhere(out==1).item()
    out[val] = out[0]
    out[0] = 1
    return out


@njit
def breed(pop, pairs):
    new = np.empty_like(pop)
    for i in range(new.shape[0]):
        splice(pop[pairs[i,0]], pop[pairs[i,1]], out=new[i])
    return new


if __name__ == "__main__":
    DATA_PATH = Path("data"); assert DATA_PATH.exists()
    NAME = "berlin52"
    #NAME = "a280"
    #NAME = "ulysses16"
    PROBLEM_PATH = DATA_PATH / f"{NAME}.tsp"
    SOLUTION_PATH = DATA_PATH / f"{NAME}.opt.tour"

    optimal = get_solutions()[NAME]
    print(NAME, optimal)

    problem = tsp.load_problem(PROBLEM_PATH)
    if problem.edge_weight_type == "GEO":
        print("WARNING:", end=' ')
        problem.wfunc = lambda i, j: \
            tsp.distances.geographical(problem.node_coords[i], problem.node_coords[j])
    print("edge weight type:", problem.edge_weight_type)
    graph = problem.get_graph()
    dmat = distance_matrix(problem.node_coords)
    s = tsp.load_solution(SOLUTION_PATH)

    nodes = np.array([*problem.get_nodes()])
    print(repr(nodes))


    n_generations = 1000000
    psize = 2000
    p = init_population(psize,nodes)
    #p[0,:] = s.tours[0]
    print(s.tours)
    new_pop = np.empty_like(p)
    ttotal = 0.0
    bestdist = float("inf")

    checkin = 200
    for i in range(n_generations+1):
        t = time()
        validate_pop(p)
        distvec = dist_vec(p, )
        besti = np.argmin(distvec)
        besttour = p[besti].copy()
        if distvec[besti] < bestdist:
            print(f"* New best  {i}  {distvec[besti]}")
            tspdraw.draw(graph, besttour, title=f"{NAME}: {bestdist}", savename=f"{NAME}/{i}.png")
            if len(besttour) < 20:
                print(besttour)
        bestdist = distvec[besti]

        if not i % checkin:
            print(
                f"Generation: {i}\t"
                + f"Best: {bestdist}\t"
                + f"Avrg: {distvec.mean():.4f}\t"
                + f"Time: {1000 * ttotal/checkin:.2f} ms\t"
            )
            ttotal = 0.0
            tspdraw.draw(graph, besttour, title=f"{NAME}: {bestdist}", savename=f"{NAME}/{i}.png")

        pdist = dist_to_fitness(distvec)
        p = breed(p, np.random.choice(psize, size=(psize,2), p=pdist))

        p[0,:] = besttour[:]  # preserve best performer
        p[1,:] = besttour[:]  # preserve best performer
        mutate(p[1])

        mutate_all(p[1:], .1)  # no mutation for best

        ttotal += time() - t


    print(p[np.argmin(distvec)])

