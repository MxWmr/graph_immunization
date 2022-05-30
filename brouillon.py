import networkx as nx
import EoN
import matplotlib.pyplot as plt
import igraph as ig
import numpy as np


N=1000
G=nx.newman_watts_strogatz_graph(N,10,0.4)

l=[]
for i in range(500):
    #PE=max_ev(G)
    PE, AR = EoN.estimate_SIR_prob_size(G, 0.3)
    l.append(PE)


var = np.mean(l)
print(var)
