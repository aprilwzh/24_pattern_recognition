#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from utils import *
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Define edit parameters
TAU = 1  # Cost of Node deletion/insertion
E_TAU = 1  # Cost of Edge deletion/insertion
K = 3  # k value, needs to be fine-tuned

# Define path to train and validation folders
train_path = 'Molecules/train.tsv'
val_path = 'Molecules/validation.tsv'

# Read in training and validation information
train_set = pd.read_csv(train_path, header=None, sep='\t')
val_set = pd.read_csv(val_path, header=None, sep='\t')

# Define functions for edit costs (to be moved to 'utils.py' once fine-tuned)
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
    e1_w, e2_w = edge1['weight'], edge2['weight']
    if e1_w != e2_w:
        return abs(e1_w - e2_w) * E_TAU
    return 0

# Convert molecules to graphs and store in train_graphs
train_graphs = []
for file_idx in train_set[0].values:
    graph_file_name = f'Molecules/gxl/{file_idx}.gxl'
    graph = parse_gxl_to_networkx(graph_file_name)
    train_graphs.append(graph)

print('Training data read in')



# Extract features from molecular graphs
def extract_features(graph):
    # Example: Extracting node and edge features
    # Modify this function based on the features you want to extract
    node_features = extract_node_features(graph)
    edge_features = extract_edge_features(graph)
    graph_features = extract_graph_features(graph)
    return node_features + edge_features + graph_features

def extract_node_features(graph):
    # Example: Extract atom type as one-hot vector
    atom_types = nx.get_node_attributes(graph, 'atom_type')
    unique_atom_types = set(atom_types.values())
    one_hot_vector = [int(atom_types[node] == atom_type) for atom_type in unique_atom_types]
    return one_hot_vector

def extract_edge_features(graph):
    # Example: Extract bond type as one-hot vector
    bond_types = nx.get_edge_attributes(graph, 'bond_type')
    unique_bond_types = set(bond_types.values())
    one_hot_vector = [int(bond_types[edge] == bond_type) for bond_type in unique_bond_types]
    return one_hot_vector

def extract_graph_features(graph):
    # Example: Extract number of nodes and edges
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    return [num_nodes, num_edges]

# Extract features from the training set
train_features = [extract_features(graph) for graph in train_graphs]
train_labels = train_set[1].values

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(train_features, train_labels)

print('Model trained')

# Make predictions on validation set
predictions = []
for file_idx in val_set[0].values:
    graph_file_name = f'Molecules/gxl/{file_idx}.gxl'
    graph = parse_gxl_to_networkx(graph_file_name)

    # Compute GED to all training data
    distances = []
    for graph2 in train_graphs:
        approximation_generator = nx.optimize_graph_edit_distance(graph, graph2,
                                                                  node_subst_cost=node_subst_cost,
                                                                  node_del_cost=node_del_cost,
                                                                  node_ins_cost=node_ins_cost,
                                                                  edge_subst_cost=edge_subst_cost,
                                                                  edge_del_cost=edge_del_cost,
                                                                  edge_ins_cost=edge_ins_cost)

        for value in approximation_generator:
            distances.append(value)
            approximation_generator.close()

    # Find closest neighbors
    min_distance_indices = np.argsort(distances)[:K]
    nearest_labels = train_set[1].values[min_distance_indices]

    # Predict based on nearest neighbors
    active_count = np.sum(nearest_labels == 'active')
    prediction = 'active' if active_count > K / 2 else 'inactive'
    predictions.append(prediction)

# Evaluate predictions
correct = np.sum(predictions == val_set[1].values)
accuracy = correct / len(val_set)
print(f'Accuracy: {accuracy}')

