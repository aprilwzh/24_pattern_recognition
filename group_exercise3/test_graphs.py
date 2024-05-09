import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import re

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
        # Extract the symbol from node attributes
        for attr in node.findall('attr'):
            if attr.attrib['name'] == 'symbol':
                symbol = attr.find('string').text.strip()
                break
        if symbol:
            graph.add_node(node_id, symbol=symbol)
            node_symbols[node_id] = symbol

    # Iterate over edges to add them to the graph
    for edge in root.findall('edge'):
        source = edge.attrib['from']
        target = edge.attrib['to']
        graph.add_edge(source, target)

    return graph, node_symbols

file_name = 'Molecules/gxl/40.gxl'
with open(file_name, 'r') as file:
    lines = file.readlines()
    gxl_str = "".join(line for line in lines if not line.startswith("<?xml"))


pattern = r'<gxl>(.*?)</gxl>'

matches = re.findall(pattern, gxl_str)
if matches:
    gxl_str = '<gxl>'+matches[0]+'</gxl>'
    # print('<gxl>'+matches[0]+'</gxl>')
else:
    print('ggrrrrr')

print('graph_string:', gxl_str)

graph, node_symbols = parse_gxl_to_networkx(gxl_str)
print("Nodes:", graph.nodes(data=True))
print("Edges:", graph.edges())

fig = plt.figure()
nx.draw(graph, labels=node_symbols, with_labels=True)
plt.show()

# generator = nx.optimize_graph_edit_distance(g2, g3)

# import networkx as nx
# import matplotlib.pyplot as plt
#
# # g1 = nx.cycle_graph(6)
# g2 = nx.wheel_graph(100)
# g3 = nx.complete_graph(64)
#
# fig = plt.figure()
#
# # nx.draw(g1)
# # plt.show()
#
# nx.draw(g2)
# plt.show()
#
# nx.draw(g3)
# plt.show()
#
# # print(g1)
# print(g2)
# print(g3)
#
# generator = nx.optimize_graph_edit_distance(g2, g3)
# print('hi')
#
# for v in generator:
#     print(v)
#     generator.close()
#
# print('done')

# import xml.etree.ElementTree as ET
# import numpy as np
# import networkx as nx
# import re
#
# G = nx.Graph()
#
# tree_gxl = ET.parse("Molecules/gxl/40.gxl")
# root_gxl = tree_gxl.getroot()
# node_id = []
# edge_attr = {}
# # Parse nodes
# for i, node in enumerate(root_gxl.iter('node')):
#     print(ET.tostring(node))
#     node_id += [node.get('id')]
#     for symbol in node:
#         print(ET.tostring(symbol))
#         print(symbol.attr)
#         break
#     break
# print(node_id)
#
# node_id = np.array(node_id)
# # Create adjacency matrix
# am = np.zeros((len(node_id), len(node_id)))
# ##Parsing edges
# for edge in root_gxl.iter('edge'):
#     s = np.where(node_id==edge.get('from'))[0][0]
#     t = np.where(node_id==edge.get('to'))[0][0]
#     break
#
#     # Get the child node of the current edge for the nrl value
#     for node in edge:
#         print(node.get('id'))
#         content = ET.tostring(node).decode("utf-8")
#         # dw(node.text)
#         # Get the nrl value via regex
#         r1 = re.findall(r"valence", content)
#         print(content)
#         print(r1)
#
#         # Modify value according to: (nlr*13.54)-13.54
#         # r1 = (float(r1[0])*13.54)-13.54
#
#     #Add edge with original node names and nlr value to graph
#     G.add_edge(node_id[s],node_id[t], nlr=r1)
#
# fig = plt.figure()
#
# nx.draw(G)
# plt.show()


# import xml.etree.ElementTree as ET
#
# def parse_gxl_node(node_str):
#     # Parse the XML string into an ElementTree object
#     root = ET.fromstring(node_str)
#
#     # Initialize variables to store extracted information
#     node_id = root.attrib['id']
#     symbol = chem = charge = x = y = None
#
#     # Iterate over child elements to extract specific attributes
#     for attr in root.findall('attr'):
#         attr_name = attr.attrib['name']
#         if attr_name == 'symbol':
#             symbol = attr.find('string').text.strip()
#         elif attr_name == 'chem':
#             chem = int(attr.find('int').text)
#         elif attr_name == 'charge':
#             charge = int(attr.find('int').text)
#         elif attr_name == 'x':
#             x = float(attr.find('float').text)
#         elif attr_name == 'y':
#             y = float(attr.find('float').text)
#
#     # Return extracted information as a dictionary
#     return {
#         'id': node_id,
#         'symbol': symbol,
#         'chem': chem,
#         'charge': charge,
#         'x': x,
#         'y': y
#     }
#
# # Example usage
# node_str = '<node id="_1"><attr name="symbol"><string>N  </string></attr><attr name="chem"><int>4</int></attr><attr name="charge"><int>0</int></attr><attr name="x"><float>4.5981</float></attr><attr name="y"><float>0.25</float></attr></node>'
# node_info = parse_gxl_node(node_str)
# print(node_info)

