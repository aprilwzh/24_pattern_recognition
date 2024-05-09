from utils import *


file_name = 'Molecules/gxl/35.gxl'

graph = parse_gxl_to_networkx(file_name)

print_graph_data(graph)
draw_graph(graph)
