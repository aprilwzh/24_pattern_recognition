import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import re
import numpy as np


def parse_gxl_to_networkx(gxl_str):
    """
    Takes file name and retrieves .gxl information, converts this to networkx graph
    :param gxl_str: file name to .gxl graph
    :return: nx.Graph
    """
    # try to extract correct portion of string, if possible, exit program safely otherwise
    try:
        gxl_str = get_xml_string(file_name=gxl_str)
    except AssertionError:
        print('Assertion Failed! Incorrect gxl format!!!')
        exit(0)
    # Parse the XML string into an ElementTree object
    root = ET.fromstring(gxl_str).findall('graph')[0]

    # Initialize a NetworkX graph
    graph = nx.Graph()

    # Iterate over nodes to extract symbols and add them as nodes to the graph
    for node in root.findall('node'):
        node_id = node.attrib['id']
        symbol = None
        # Create variable to store drawing position
        pos = np.zeros(2)
        # Extract the symbol from node attributes
        for attr in node.findall('attr'):
            if attr.attrib['name'] == 'symbol':
                symbol = attr.find('string').text.strip()
            if attr.attrib['name'] == 'x':  # get 'x' position for drawing
                pos[0] = float(attr.find('float').text)
            if attr.attrib['name'] == 'y':  # get 'y' position for drawing
                pos[1] = float(attr.find('float').text)
        if symbol:  # Add Atom label and position
            graph.add_node(node_id, symbol=symbol, pos=pos)  # Add node to graph

    # Iterate over edges to add them to the graph
    for edge in root.findall('edge'):
        # Set default weight
        weight = 1
        for attr in edge.findall('attr'):
            if attr.attrib['name'] == 'valence':  # Treat valence as edge weight, indicates bond strength
                weight = int(attr.find('int').text)
        source = edge.attrib['from']
        target = edge.attrib['to']
        graph.add_edge(source, target, weight=weight)  # Add edge to graph

    return graph


def get_xml_string(file_name):
    """
    Parse exact graph portion of .gxl file, ignores irrelevant headers/extra information
    :param file_name: File name of .gxl file
    :return: extracted string
    :raises: AssertionError if no correct file information exists
    """
    # Get initial file information
    with open(file_name, 'r') as file:
        lines = file.readlines()
        gxl_str = "".join(line for line in lines if not line.startswith("<?xml"))

    # Define correct graph pattern
    pattern = r'<gxl>(.*?)</gxl>'

    # Check for matches in file contents
    matches = re.findall(pattern, gxl_str)
    if matches:
        # Ensure correct format
        gxl_str = '<gxl>' + matches[0] + '</gxl>'
    else:
        print('Warning! No match found...')
        raise AssertionError  # correct information not found

    return gxl_str


def print_graph_data(g):
    """
    Prints node and edge data
    :param g: graph
    :return: None
    """
    print("Nodes:", g.nodes(data=True))
    print("Edges:", g.edges(data=True))


def draw_graph(g, i=None):
    """
    Draws graph G
    :param g: graph
    :param i: file index for main title of plot (optional)
    :return: None
    """
    # Collect node positions
    pos = nx.get_node_attributes(g, 'pos')

    # Create figure (can be used if you want to save figures)
    fig = plt.figure()
    # Draw graph with positions and Atom symbols
    nx.draw(g, pos=pos, labels=nx.get_node_attributes(g, 'symbol'), with_labels=True)
    # Get edge labels
    labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)  # Add edge labels to graph

    # Check to see if you need to add a plot name
    if i is not None:
        plt.suptitle(f'file_number: {i}')
        print(f'file_number: {i}')
    # Display graph
    plt.show()
