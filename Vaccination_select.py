import numpy as np
import retworkx as rx
import networkx as nx 
from tqdm import tqdm



# select the M nodes with highest degree for vaccination
def deg_max(G,M): 
    vaccinated=[]
    # degree of all nodes sorted by index of the nodes
    degree_sequence = [val for (node, val) in sorted(G.degree(), key=lambda pair: pair[0])]
    degree_sequence2 = degree_sequence.copy()
    for i in range(M):
        maxi=max(degree_sequence2)
        vaccinated.append(degree_sequence.index(maxi))
        degree_sequence[degree_sequence.index(maxi)]=0
        degree_sequence2.remove(maxi)

    return vaccinated


# selection for vacc by max betweenness centrality without recomputation of centrality
def centrality_max_no_recomp(G,M): 
    vaccinated=[]
    # degree of all node sorted by index of the nodes
    centrality_sequence = list(nx.betweenness_centrality(G).values())
    centrality_sequence2 = centrality_sequence.copy()
    for i in range(M):
        maxi=max(centrality_sequence2)
        vaccinated.append(centrality_sequence.index(maxi))
        centrality_sequence[centrality_sequence.index(maxi)]=-10
        centrality_sequence2.remove(maxi)

    return vaccinated

# selection for vacc by max betweenness centrality with recomputation of centrality
def centrality_max_recomp(G,M): 

    G2 = rx.networkx_converter(G)
    vaccinated=[]
    # degree of all node sorted by index of the nodes
    dico=rx.betweenness_centrality(G2)
    centrality_sequence = list(dico.values())
    for i in tqdm(range(M)):
        maxi=max(centrality_sequence)
        node=[k for k in dico.keys() if dico[k] == maxi][0]
        vaccinated.append(node)
        G2.remove_node(node)
        dico=rx.betweenness_centrality(G2)
        centrality_sequence = list(dico.values())

    return vaccinated

#G = nx.newman_watts_strogatz_graph(1000,10,0.4)
#v=centrality_max_recomp(G,300)
#print(v)