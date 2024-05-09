from utils import *
import pandas as pd

# DEFINE EDIT PARAMETERS
TAU = 1  # Cost of Node deletion/insertion
E_TAU = 1  # Cost of Edge deletion/insertion
K = 3  # k value, needs to be fine tuned


# TODO: Move following functions into 'utils.py' once parameters (TAU and E_TAU) have been fine-tuned
def node_del_cost(node):
    return TAU


def node_ins_cost(node):
    return TAU


def node_subst_cost(node1, node2):
    if node1['symbol'] != node2['symbol']:  # if Atom labels !=, return cost of deletion + insertion
        return 2 * TAU
    return 0  # If Atom label ==, return 0 cost


def edge_del_cost(edge):
    # Way to include the valence of a bond (e.g. if valence = 2, would cost more energy to break than valence = 1)...
    # return E_TAU * edge['weight']  # Doesn't immediately add much value
    return E_TAU


def edge_ins_cost(edge):
    # Way to include the valence of a bond (e.g. if valence = 2, would cost more energy to break than valence = 1)...
    # return E_TAU * edge['weight']  # Doesn't immediately add much value
    return E_TAU


def edge_subst_cost(edge1, edge2):
    e1_w, e2_w = edge1['weight'], edge2['weight']
    if e1_w != e2_w:  # if valence !=, return Edge edit cost * difference
        return abs(e1_w - e2_w) * E_TAU
    return 0  # If valence ==, return 0 cost


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

    # Useful for visualizing graph information!
    # print_graph_data(graph)
    # if i % 25 == 0:
    #     draw_graph(graph, file_idx)

# Notify user of progress
print('training data read in')

# Define path to validation folder
val_path = 'Molecules/validation.tsv'
# Read in validation information
val_set = pd.read_csv(val_path, header=None, sep='\t')

predictions = []
# Iterate through validation date, compute GED to all training data, find closest neighbors, store prediction
for i, file_idx in enumerate(val_set[0].values):
    # Collect file name
    graph_file_name = f'Molecules/gxl/{file_idx}.gxl'
    # Get graph from utils.py function
    graph = parse_gxl_to_networkx(graph_file_name)

    # Create distance array
    values = np.zeros(len(train_set))
    # Iterate through all training graphs, compute GED
    for j, graph2 in enumerate(train_graphs):
        # Create Approximate GED generator with pre-defined node & edge cost functions from above
        approximation_generator = nx.optimize_graph_edit_distance(graph, graph2,
                                                                  node_subst_cost=node_subst_cost,
                                                                  node_del_cost=node_del_cost,
                                                                  node_ins_cost=node_ins_cost,
                                                                  edge_subst_cost=edge_subst_cost,
                                                                  edge_del_cost=edge_del_cost,
                                                                  edge_ins_cost=edge_ins_cost)

        # Retrieve value from first approximation (upper bound) of GED
        for value in approximation_generator:
            values[j] = value  # Store GED
            approximation_generator.close()  # stop generator to ignore exact edit cost & save time

    # Sort GED values
    min_value_idxs = np.argsort(values)[:K]
    min_values = values[min_value_idxs]

    # Get closest training classes
    active_inactive = train_set[1].values[min_value_idxs]
    active_inactive_idxs = train_set[0].values[min_value_idxs]  # Get file names for debugging/checking if needed

    # Check if active or inactive based off of closest predictions
    c = 0
    for val in active_inactive:
        if val == 'active':
            c += 1

    if c > np.floor(K/2):
        predictions.append('active')
    else:
        predictions.append('inactive')

    # Print debugging information
    # print(i, min_values, active_inactive, active_inactive_idxs, 'gt:', val_set.loc[i, 1])

    # Provide progress update to user
    print('step:', i)

# Count correct predictions
correct = 0
for i in range(len(val_set)):
    if predictions[i] == train_set[1].values[i]:
        correct += 1

print(f'number of correct predictions: {correct}')
