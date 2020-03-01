import numpy as np
import dask
from tqdm import tqdm
from numba import jit, njit, prange
from random import random
from time import time


def side_by_side(a, b):
    for n,(i,j) in enumerate(zip(a,b)):
        print(f"{n}  {i:.5f}   {j:.5f}   {i-j:.5f}")


def init_population(n,m):
    return np.random.random((n,m))


class mutation_tracker:
    total = 0
    count = 0

    @staticmethod
    def compute():
        if mutation_tracker.count:
            return mutation_tracker.total / mutation_tracker.count
        return None


@njit
def mutate(a, sigma=.1):
    #while random() > .5:
        size = int(random() * len(a))
        modifier = np.random.randn() * sigma
        a[size] += modifier
        a[size] = abs(a[size]) % 1.
    #a[int(random() * len(a))] = random()


@njit
def mutate_all(pop, prob, sigma=1):
    for i in range(pop.shape[0]):
        if prob < random():
            mutate(pop[i], sigma)


@njit#(parallel=True)
def dist_vec(p):
    pdist = np.empty(p.shape[0])
    for i in range(p.shape[0]):  # prange(p.shape[0]):
        #pdist[i] = np.linalg.norm(p[i] - sol)
        pdist[i] = np.prod(np.sin(4*np.pi*p[i])) * np.prod(np.exp(p[i]))
        #if pdist[i] < threshold: done = True
    return pdist


@njit
def dist_to_fitness(vec):
    pdist = vec / vec.max()
    #pdist = 1 - pdist
    pdist **= 2
    #pdist += 1e-10
    pdist /= pdist.sum()
    return pdist


@njit
def splice(a,b):
    spl = int(random() * len(a))
    offspring = a.copy()  # np.empty_like(a)
    #offspring[:spl] = a[:spl]
    offspring[spl:] = b[spl:]
    return offspring


@njit
def breed(pop, pairs):
    new = np.empty_like(pop)
    for i in range(new.shape[0]):
        new[i,:] = splice(pop[pairs[i,0]], pop[pairs[i,1]])[:]
    return new


@dask.delayed(pure=True)
@jit(forceobj=True)
def next_generation(p):
    distvec = dist_vec(p)
    pdist = dist_to_fitness(distvec)

    bestpoint = p[np.argmax(pdist)].copy()
    p = breed(p, np.random.choice(psize, p=pdist, size=(psize,2)))

    p[0,:] = bestpoint[:]  # preserve best performer
    p[1,:] = np.mean(p, axis=0)

    mutate_all(p[1:], .4, sigma=.1)

    return p#, (bestpoint, bestdist)


@dask.delayed(pure=True)
def best_of(p):
    return np.min(dist_vec(p))


def merge_subs(plist, bests):
    for p in plist:
        for i in range(len(bests)):
            if random() < .5:
                p[i] = bests[i]


solsize = 32
n_generations = 512
n_populations = 8
psize = 512
plist = [init_population(psize,solsize) for _ in range(n_populations)]
merge = n_generations // 4
ttotal = 0.0

for i in tqdm(range(n_generations)):

    for k in range(n_populations):
        plist[k] = next_generation(plist[k])

    if not i % merge:
        plist = [*dask.compute(*plist)]
        bests = dask.compute(*map(best_of, plist))
        merge_subs(plist, bests)

    #if not i % 2000:
    #    print(
    #        f"Generation: {i}\t"
    #        + f"Best: {distvec.min()}\t"
    #        + f"Avrg: {distvec.mean():.4f}\t"
    #        + f"Time: {1000 * ttotal/2000:.2f} ms\t"
    #    )
    #    ttotal = 0.0

plist = [*dask.compute(*plist)]

full_list = []
for p in plist:
    for i in p:
        full_list.append(i)
full_list = np.array(full_list)
distvec = dist_vec(full_list)
besti = np.argmax(distvec)
print(distvec[besti])
print(full_list[besti])

