from utils import *
import pandas as pd
import numpy as np
import networkx as nx

# DEFINE EDIT PARAMETERS
TAU = 1  # Cost of Node deletion/insertion
E_TAU = 1  # Cost of Edge deletion/insertion
K = 3  # k value, needs to be fine tuned

# Define cost functions
def node_del_cost(node):
    return TAU

def node_ins_cost(node):
    return TAU

def node_subst_cost(node1, node2):
    if node1['symbol'] != node2['symbol']:
        return 2 * TAU
    return 0

def edge_del_cost(edge):
    return E_TAU

def edge_ins_cost(edge):
    return E_TAU

def edge_subst_cost(edge1, edge2):
    if edge1['weight'] != edge2['weight']:
        return abs(edge1['weight'] - edge2['weight']) * E_TAU
    return 0

# Define path to train folder
train_path = 'Molecules/train.tsv'
# Read in training information
train_set = pd.read_csv(train_path, header=None, sep='\t')

train_graphs = []
# Iterate through training data, convert to networkx graph, store graph
for i, file_idx in enumerate(train_set[0].values):
    # Collect file name
    graph_file_name = f'Molecules/gxl/{file_idx}.gxl'
    # Get graph from utils.py function
    graph = parse_gxl_to_networkx(graph_file_name)
    train_graphs.append(graph)

print('training data read in')

# Define path to validation folder
val_path = 'Molecules/validation.tsv'
# Read in validation information
val_set = pd.read_csv(val_path, header=None, sep='\t')

predictions = []
# Iterate through validation data, compute GED to all training data, find closest neighbors, store prediction
for i, file_idx in enumerate(val_set[0].values):
    graph_file_name = f'Molecules/gxl/{file_idx}.gxl'
    graph = parse_gxl_to_networkx(graph_file_name)
    values = np.zeros(len(train_set))
    for j, graph2 in enumerate(train_graphs):
        approximation_generator = nx.optimize_graph_edit_distance(graph, graph2,
                                                                  node_subst_cost=node_subst_cost,
                                                                  node_del_cost=node_del_cost,
                                                                  node_ins_cost=node_ins_cost,
                                                                  edge_subst_cost=edge_subst_cost,
                                                                  edge_del_cost=edge_del_cost,
                                                                  edge_ins_cost=edge_ins_cost)
        for value in approximation_generator:
            values[j] = value
            approximation_generator.close()
    min_value_idxs = np.argsort(values)[:K]
    active_inactive = train_set[1].values[min_value_idxs]
    c = np.sum(active_inactive == 'active')
    if c > np.floor(K / 2):
        predictions.append('active')
    else:
        predictions.append('inactive')
    print('step:', i)

correct = 0
for i in range(len(val_set)):
    if predictions[i] == train_set[1].values[i]:
        correct += 1
print(f'number of correct predictions: {correct}')

# Path to the new test data
test_path = 'Molecules-test/test.tsv'

def read_graphs(file_path):
    dataset = pd.read_csv(file_path, header=None, sep='\t')
    graphs = []
    for file_idx in dataset[0].values:
        graph_file_name = f'Molecules-test/gxl/{file_idx}.gxl'
        graph = parse_gxl_to_networkx(graph_file_name)
        graphs.append(graph)
    return dataset, graphs

test_set, test_graphs = read_graphs(test_path)

def predict_classes(graphs_to_predict, reference_graphs, reference_labels, K):
    predictions = []
    for graph in graphs_to_predict:
        values = np.zeros(len(reference_graphs))
        for j, ref_graph in enumerate(reference_graphs):
            approximation_generator = nx.optimize_graph_edit_distance(graph, ref_graph,
                                                                      node_subst_cost=node_subst_cost,
                                                                      node_del_cost=node_del_cost,
                                                                      node_ins_cost=node_ins_cost,
                                                                      edge_subst_cost=edge_subst_cost,
                                                                      edge_del_cost=edge_del_cost,
                                                                      edge_ins_cost=edge_ins_cost)
            for value in approximation_generator:
                values[j] = value
                approximation_generator.close()
        min_value_idxs = np.argsort(values)[:K]
        active_inactive = reference_labels[min_value_idxs]
        c = np.sum(active_inactive == 'active')
        if c > np.floor(K / 2):
            predictions.append('active')
        else:
            predictions.append('inactive')
    return predictions

test_predictions = predict_classes(test_graphs, train_graphs, train_set[1].values, K)

# Print test predictions (if ground truth is not available)
print('Test predictions:', test_predictions)

# Save predictions to a TSV file
output_path = 'test_predictions.tsv'
with open(output_path, 'w') as f:
    for idx, label in zip(test_set[0].values, test_predictions):
        f.write(f"{idx}\t{label}\n")

print(f'Test predictions saved to {output_path}')
