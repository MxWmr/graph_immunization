import networkx as nx
import EoN
import matplotlib.pyplot as plt
import igraph as ig



def plot_graph(G,name=None):
    if name is not None:
        nx.write_gml(G,name+'.gml')

    g = ig.Graph.from_networkx(G)

    visual_style = {}

    # Define colors used for outdegree visualization
    colours = ['#fecc5c', '#a31a1c']

    # Set bbox and margin
    visual_style["bbox"] = (3000,3000)
    visual_style["margin"] = 17

    # Set vertex colours
    visual_style["vertex_color"] = 'grey'

    # Set vertex size
    visual_style["vertex_size"] = 20

    # Set vertex lable size
    visual_style["vertex_label_size"] = 8

    # Don't curve the edges
    visual_style["edge_curved"] = False

    # Set the layout
    my_layout = g.layout_fruchterman_reingold()
    visual_style["layout"] = my_layout

    # Plot the graph
    ig.plot(g, **visual_style)



def save_graph(G,name):
    nx.write_gml(G,'saved_graph/'+name+'.gml')


#G = nx.newman_watts_strogatz_graph(1000,10,0.4)
#plot_graph(G)