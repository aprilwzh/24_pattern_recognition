import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import re
import numpy as np


def parse_gxl_to_networkx(gxl_str):
    # Parse the XML string into an ElementTree object
    root = ET.fromstring(gxl_str).findall('graph')[0]
    # print(root.findall('graph')[0].findall('node'))

    # Initialize a NetworkX graph
    graph = nx.Graph()
    node_symbols = {}

    # Iterate over nodes to extract symbols and add them as nodes to the graph
    for node in root.findall('node'):
        node_id = node.attrib['id']
        symbol = None
        pos = np.zeros(2)
        # Extract the symbol from node attributes
        for attr in node.findall('attr'):
            if attr.attrib['name'] == 'symbol':
                symbol = attr.find('string').text.strip()
            if attr.attrib['name'] == 'x':
                pos[0] = float(attr.find('float').text)
            if attr.attrib['name'] == 'y':
                pos[1] = float(attr.find('float').text)
        if symbol:
            graph.add_node(node_id, symbol=symbol, pos=pos)
            # node_symbols[node_id] = symbol

    # Iterate over edges to add them to the graph
    for edge in root.findall('edge'):
        weight = 1
        for attr in edge.findall('attr'):
            if attr.attrib['name'] == 'valence':
                weight = int(attr.find('int').text)
        source = edge.attrib['from']
        target = edge.attrib['to']
        graph.add_edge(source, target, weight=weight)

    return graph


def get_xml_string(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        gxl_str = "".join(line for line in lines if not line.startswith("<?xml"))

    pattern = r'<gxl>(.*?)</gxl>'

    matches = re.findall(pattern, gxl_str)
    if matches:
        gxl_str = '<gxl>' + matches[0] + '</gxl>'
    else:
        print('Warning! No match found...')
        raise AssertionError

    return gxl_str


def print_graph_data(g):
    print("Nodes:", g.nodes(data=True))
    print("Edges:", g.edges(data=True))


def draw_graph(g):
    pos = nx.get_node_attributes(g, 'pos')

    fig = plt.figure()
    nx.draw(g, pos=pos, labels=nx.get_node_attributes(g, 'symbol'), with_labels=True)
    labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

    plt.show()
