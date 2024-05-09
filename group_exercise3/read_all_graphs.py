import networkx as nx
import numpy as np

from utils import *
import pandas as pd

train_path = 'Molecules/train.tsv'
train_set = pd.read_csv(train_path, header=None, sep='\t')

train_graphs = []

for i, file_idx in enumerate(train_set[0].values):
    graph_file_name = f'Molecules/gxl/{file_idx}.gxl'

    graph = parse_gxl_to_networkx(graph_file_name)

    # print_graph_data(graph)
    # if i % 25 == 0:
    #     draw_graph(graph, file_idx)

    train_graphs.append(graph)

print('training data read in')

val_path = 'Molecules/validation.tsv'
val_set = pd.read_csv(val_path, header=None, sep='\t')

for i, file_idx in enumerate(val_set[0].values):
    graph_file_name = f'Molecules/gxl/{file_idx}.gxl'

    graph = parse_gxl_to_networkx(graph_file_name)

    values = np.zeros(len(train_set))
    for j, graph2 in enumerate(train_graphs):
        approximation_generator = nx.optimize_graph_edit_distance(graph, graph2)

        for value in approximation_generator:
            values[j] = value
            approximation_generator.close()

    min_value_idxs = np.argsort(values)[:3]
    min_values = values[min_value_idxs]
    active_inactive = train_set[1].values[min_value_idxs]
    print(i, min_values, active_inactive, 'gt:', val_set.loc[i, 1])
