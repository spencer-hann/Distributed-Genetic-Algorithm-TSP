import numpy as np
import matplotlib.pyplot as plt
import tsplib95 as tsp
import networkx as nx

from pathlib import Path


class EdgeSet(set):
    def __contains__(self, item):
        # item must be tuple
        return super().__contains__(item) or \
                super().__contains__((item[1], item[0]))


def tour_to_edgeset(tour):
    s = EdgeSet()
    for i in range(len(tour)-1):
        s.add((tour[i], tour[i+1]))
    return s


def draw(
    graph,
    tour=tuple(),
    title='',
    default='k',
    highlight='r',
    savename="graph.png",
    #figsize=(16,16),
    label_edges=False,
):
    pos = {n:d['coord'] for n, d in graph.nodes.items()}
    #pos = nx.spring_layout(graph)
    edgelist = tour_to_edgeset(tour)

    nx.draw_networkx(graph, pos, with_labels=True, width=.4)
    nx.draw_networkx_edges(graph, pos, edgelist=edgelist, width=2, edge_color='r')

    if label_edges:
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=nx.get_edge_attributes(graph, "weight"))

    #plt.figure(figsize=figsize)
    plt.title(title)
    plt.savefig(savename)
    plt.cla()


if __name__ == "__main__":

    DATA_PATH = Path("data"); assert DATA_PATH.exists()
    NAME = "berlin52"
    NAME = "ulysses16"
    PROBLEM_PATH = DATA_PATH / f"{NAME}.tsp"
    SOLUTION_PATH = DATA_PATH / f"{NAME}.opt.tour"

    p = tsp.load_problem(PROBLEM_PATH)
    s = tsp.load_solution(SOLUTION_PATH)

    print(f"Optimal tour: {s.tours}")
    print(f"Optimal tour: {p.trace_tours(s)}")
    g = p.get_graph()
    draw(g, s.tours[0])


#def get_colors(
#    graph, edges, default='b', highlight='r',
#):
#    c = np.full(len(graph.edges), default)
#    w = np.ones(len(graph.edges))
#
#    for i, edge in enumerate(graph.edges):
#        if edge in edges:
#            c[i] = highlight
#            w[i] = 4.0
#
#    return c, w
