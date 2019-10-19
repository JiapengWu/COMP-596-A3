from util import *
import matplotlib.pyplot as plt
import pdb
from networkx.algorithms.community import LFR_benchmark_graph
from networkx.algorithms import node_classification
import numpy as np
from sklearn.metrics import accuracy_score


def drop_label(graph, labels, percentage):

    nodes = graph.nodes
    indices = np.random.permutation(len(nodes))
    bound = int(len(nodes) * percentage)
    training_idx, test_idx = indices[:bound], indices[bound:]
    train_mask = np.zeros(len(nodes))
    train_mask[training_idx] = 1

    for node, label, mask in zip(nodes, labels, train_mask):
        if mask == 1:
            graph.nodes[node]['label'] = label
        else:
            try:
                nodes[node].pop("label")
            except:
                pass

    harmonic_prediction = np.array(node_classification.harmonic_function(graph))
    local_global_prediction = np.array(node_classification.local_and_global_consistency(graph))

    return accuracy_score(labels[test_idx], harmonic_prediction[test_idx]), \
           accuracy_score(labels[test_idx], local_global_prediction[test_idx])


def q1(graph):
    nodes = graph.nodes
    if args.data == 'karate':
        clubs = list(set([graph.nodes[x]['club'] for x in nodes]))
        labels = [clubs.index(graph.nodes[x]['club']) for x in nodes]
    else:
        labels = [graph.nodes[x]['value'] for x in nodes]

    labels = np.array(labels)
    harmonic = []
    loc_glob = []
    for percentage in np.arange(0.05, 1, 0.05):

        harmonic_accs = []
        local_global_accs = []
        print('Dropping {:.3f} percent nodes labels...'.format(1 - percentage))
        for i in range(10):
            harmonic_acc, local_global_acc = drop_label(graph.copy(), labels, percentage)
            harmonic_accs.append(harmonic_acc); local_global_accs.append(local_global_acc)
        harmonic.append(np.mean(harmonic_accs))
        loc_glob.append(np.mean(local_global_accs))
        # print("Harmonic acc: {:.4f}".format(np.mean(harmonic_accs)) * 100)
        # print("Loc-glob acc: {:.4f}".format(np.mean(local_global_accs)) * 100)

    for i in harmonic:
        print("{:.4f}".format(i))
    print()
    for i in loc_glob:
        print("{:.4f}".format(i))


def q2():
    graph, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_gcn(args.data)
    if args.data == 'citeseer':
        nonzeros = labels.nonzero()
        max_label = np.max(nonzeros[1])
        diff = len(labels) - len(nonzeros[1])
        res = np.full(labels.shape[0], -1)
        res[nonzeros[0]] = nonzeros[1]
        iso_mask = np.where(res == -1)
        res[iso_mask] = np.arange(max_label + 1, max_label + 1 + diff)
        labels = res
    else:
        labels = labels.nonzero()[1]

    for node, label, mask in zip(graph.nodes, labels, train_mask):
        if mask:
            graph.nodes[node]['label'] = label

    harmonic_prediction = np.array(node_classification.harmonic_function(graph))
    local_global_prediction = np.array(node_classification.local_and_global_consistency(graph))

    print("train accuracy of harmonic prediction: {}".format(accuracy_score(labels[train_mask], harmonic_prediction[train_mask])))
    print("train accuracy of loc-glob prediction: {}".format(accuracy_score(labels[train_mask], local_global_prediction[train_mask])))

    print("val accuracy of harmonic prediction: {}".format(accuracy_score(labels[val_mask], harmonic_prediction[val_mask])))
    print("val accuracy of loc-glob prediction: {}".format(accuracy_score(labels[val_mask], local_global_prediction[val_mask])))

    print("test accuracy of harmonic prediction: {}".format(accuracy_score(labels[test_mask], harmonic_prediction[test_mask])))
    print("test accuracy of loc-glob prediction: {}".format(accuracy_score(labels[test_mask], local_global_prediction[test_mask])))


def q3():
    harmonic = []
    loc_glob = []
    for mu in np.arange(0.1, 1, 0.1):
        graph = LFR_benchmark_graph(1000, 3, 1.5, mu=mu, min_community=20, average_degree=5, seed=3)
        nodes = graph.nodes
        communities = {frozenset(graph.nodes[v]['community']) for v in graph}
        res = np.zeros(len(nodes)).astype(int)
        for i, part in enumerate(communities):
            for j in part:
                res[j] = i

        harmonic_accs = []
        local_global_accs = []

        for i in range(10):
            harmonic_acc, local_global_acc = drop_label(graph.copy(), res, 0.8)
            harmonic_accs.append(harmonic_acc);local_global_accs.append(local_global_acc)

        harmonic.append(np.mean(harmonic_accs))
        loc_glob.append(np.mean(local_global_accs))

    for i in harmonic:
        print("{:.4f}".format(i))
    print()
    for i in loc_glob:
        print("{:.4f}".format(i))


if __name__ == '__main__':
    args = get_args()

    if args.data in q1_data:
        if args.data == 'karate':
            graph = nx.karate_club_graph()
        else:
            graph = nx.read_gml("real-classic/{}.gml".format(args.data), label='id').to_undirected()
        q1(graph)

    elif args.data == 'LFR':
        q3()

    else:
        q2()
