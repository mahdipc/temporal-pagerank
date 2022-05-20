__author__ = 'Polina (edited by MahDi Molavi)'
import networkx as nx
import copy
import numpy as np
from datetime import datetime, timedelta
import os.path
import matplotlib.pyplot as plt


def getToy():
    #G = nx.Graph()
    #G.add_edges_from([(1,2,{'weight': 0.25}), (2,3, {'weight': 0.25})])
    G = nx.DiGraph()
    G.add_edges_from([(1, 2, {'weight': 1.0}), (3, 2, {'weight': 1.0})])
    nrm = float(sum(G.degree(weight='weight').values()))
    for i in G.edges(data=True):
        G[i[0]][i[1]]['weight'] = i[-1]['weight']/nrm
    return G


def getSubgraph(G, N):
    Gcc = sorted([G.subgraph(c) for c in nx.connected_components(
        G.to_undirected())], key=len, reverse=True)
    print(len(Gcc))
    nodes = set()
    i = 0

    while len(nodes) < N:
        s = np.random.choice(Gcc[i].nodes())
        i += 1
        nodes.add(s)
        for edge in nx.bfs_edges(G.to_undirected(), s):
            nodes.add(edge[1])
            if len(nodes) == N:
                break
    return nx.subgraph(G, nodes)


def getGraph(edgesTS):
    G = nx.DiGraph()
    edges = {}

    for item in edgesTS:
        edge = item[1]
        edges[edge] = edges.get(edge, 0.0) + 1.0

    #nrm = float(sum(edges.values()))
    G.add_edges_from([(k[0], k[1], {'weight': v}) for k, v in edges.items()])
    # G.add_edges_from([tuple(edge)])
    return G


def readRealGraph(filepath):
    edgesTS = []
    nodes = set()
    edges = set()
    lookup = {}

    weights = {}
    c = 0
    with open(filepath, 'r') as fd:
        for line in fd.readlines():

            line = line.strip()
            items = line.split(' ')
            tstamp = ' '.join(items[0:2])
            tstamp = tstamp[1:-1]
            tstamp = datetime.strptime(tstamp, '%Y-%m-%d %H:%M:%S')

            t = items[2:4]
            t = list(map(int, t))

            t[0] += 1
            t[1] += 1
            if t[0] == t[1]:
                continue
            # t.sort(); #undirected

            if tuple(t) in lookup.keys():
                num = lookup[tuple(t)]
            else:
                num = c
                lookup[tuple(t)] = c
                c += 1

            edgesTS.append((tstamp, tuple(t), num))
            nodes.add(t[0])
            nodes.add(t[1])
            edges.add(tuple([t[0], t[1]]))
            # weights[str(t[0])+','+str(t[1])] = int(items[4])
    fd.close()
    return edgesTS, nodes, edges, weights


def weighted_DiGraph(dir):

    edgesTS, nodes, edges, weights_input = readRealGraph(dir)
    G = getGraph(edgesTS)
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = getSubgraph(G, len(nodes))

    G = G.copy()

    for i in G.nodes():
        if G.out_degree(i) == 0:
            for j in G.nodes():
                if i != j:
                    G.add_edge(i, j, weight=1)

    print(nx.info(G))
    nrm = float(sum([val for (node, val) in G.degree()]))

    # nrm = float(sum(G.out_degree(weight='weight').values()))
    for i in G.edges(data=True):
        G[i[0]][i[1]]['weight'] = i[-1]['weight']/nrm
    return G


def change_weights(G):
    #w = np.random.uniform(1e-5, 1.0, G.number_of_edges())
    w = np.random.uniform(0.0, 1.0, G.number_of_edges())
    w /= sum(w)
    c = 0
    for i in G.edges():
        G[i[0]][i[1]]['weight'] = w[c]
        c += 1
    return G
