from util import *
import matplotlib.pyplot as plt
import pdb
from networkx.algorithms.community import LFR_benchmark_graph
from networkx.algorithms import node_classification

def load_labels(graph):
    # sorted_nodes = sorted(list(graph.nodes), key=lambda s: int(s))
    global labels
    sorted_nodes = graph.nodes

    if args.data in q1_data:
        if args.data == 'karate':
            clubs = list(set([graph.nodes[x]['club'] for x in sorted_nodes]))
            labels = [clubs.index(graph.nodes[x]['club']) for x in sorted_nodes]
        else:
            labels = [graph.nodes[x]['value'] for x in sorted_nodes]

        for node, label in zip(graph.nodes, labels):
            graph.nodes[node]['label'] = label
        pdb.set_trace()

    elif args.data == 'LFR':
        communities = {frozenset(graph.nodes[v]['community']) for v in graph}
        res = np.zeros(len(sorted_nodes)).astype(int)
        for i, part in enumerate(communities):
            for j in part:
                res[j] = i
        labels = res.tolist()

        for node, label in zip(graph.nodes, labels):
            node['label'] = label
    else:
        if args.data == 'citeseer':
            nonzeros = labels.nonzero()
            max_label = np.max(nonzeros[1])
            diff = len(labels) - len(nonzeros[1])
            res = np.full(labels.shape[0], -1)
            res[nonzeros[0]] = nonzeros[1]
            iso_mask = np.where(res == -1)
            res[iso_mask] = np.arange(max_label + 1, max_label + 1 + diff)
            labels = res.tolist()
        else:
            labels = labels.nonzero()[1].tolist()
    return labels



if __name__ == '__main__':

    args = get_args()

    if args.data in q1_data:
        if args.data == 'karate':
            graph = nx.karate_club_graph()
        else:
            graph = nx.read_gml("real-classic/{}.gml".format(args.data), label='id').to_undirected()

        labels = load_labels(graph)
        node_classification.harmonic_function(graph)

        pdb.set_trace()


    elif args.data == 'LFR':
        graph = LFR_benchmark_graph(200, 2.5, 1.5, 0.1, min_community=10, min_degree=5, seed=10)
    else:
        graph, labels = load_data_gcn(args.data)

    pos = nx.spring_layout(graph)
