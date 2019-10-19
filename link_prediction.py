from util import *
import pdb
from networkx.algorithms.community import LFR_benchmark_graph
from networkx.algorithms import link_prediction
import numpy as np
from sklearn.metrics import roc_auc_score
import networkx as nx


def convert_multigraph_to_single_graph(graph):
    edges = [(u,v) for u,v,k in graph.edges]
    # nodes = graph.copy().nodes
    graph = nx.Graph(edges)

    # non_zero = [(u,v,k) for u,v,k in graph.edges if k != 0]
    # zeros = [(u,v,k) for u,v,k in graph.edges if k == 0]
    # for u,v,k in non_zero:
    #     graph.remove_edge(u,v,k)
    # for u,v,k in zeros:
    #     graph.remove_edge(u,v,k)
    #     pdb.set_trace()
    #     graph.add_edge(u,v)
    return graph


def drop_edge(graph):
    edges = graph.edges
    indices = np.random.permutation(len(edges))
    bound = int(len(edges) * 0.8)
    training_idx, test_idx = indices[:bound], indices[bound:]

    test_edges = np.array(edges)[test_idx].tolist()
    # non_zero = [(u,v,k) for u,v,k in edges if k != 0]
    # self_loop = [(u,v) for u,v,k in edges if u == v]
    # pdb.set_trace()
    for u,v in test_edges:
        graph.remove_edge(u,v)

    pred_adamic = list(link_prediction.adamic_adar_index(graph))

    unpre_edges = [(u,v) for u, v, p in pred_adamic]
    score_adamic = [ p for u, v, p in pred_adamic]
    pred_jaccard = list(link_prediction.jaccard_coefficient(graph))
    score_jaccard = [ p for u, v, p in pred_jaccard]
    label = [1 if [u, v] in test_edges else 0 for u,v in unpre_edges]

    adamic_results = roc_auc_score(label, score_adamic)
    jaccard_results = roc_auc_score(label, score_jaccard)

    return adamic_results, jaccard_results


def compute_avg_auc(graph):
    adamic_results = []
    jaccard_results = []
    for i in range(10):
        adamic_result, jaccard_result = drop_edge(graph.copy())
        adamic_results.append(adamic_result)
        jaccard_results.append(jaccard_result)
    # print("Average AUC value of Adamic Adar index: {}".format(np.mean(adamic_results)))
    # print("Average AUC value of jaccard coefficient: {}".format(np.mean(jaccard_results)))
    print("{:.4f}".format(np.mean(adamic_results)))
    print("{:.4f}".format(np.mean(jaccard_results)))
    # print("Average AUC value of jaccard coefficient: {}".format(np.mean(jaccard_results)))


def q2():
    graph, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_gcn(args.data)
    compute_avg_auc(graph)


if __name__ == '__main__':
    args = get_args()
    if args.data in q1_data:
        if args.data == 'karate':
            graph = nx.karate_club_graph()
        else:
            graph = nx.read_gml("real-classic/{}.gml".format(args.data), label='id').to_undirected()
            if args.data != 'strike':
                graph = convert_multigraph_to_single_graph(graph)
        graph = compute_avg_auc(graph)
        # pdb.set_trace()
    elif args.data == 'LFR':

        for mu in np.arange(0.1, 1, 0.1):
            graph = LFR_benchmark_graph(1000, 3, 1.5, mu=mu, min_community=20, average_degree=5, seed=3)
            compute_avg_auc(graph)
    else:
        q2()

