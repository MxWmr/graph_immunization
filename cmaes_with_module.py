import numpy as np
import cma
import random as rd
import networkx as nx
import retworkx as rx
from vulnerability_meas import max_ev

cma.CMAOptions('tol')  # display 'tolerance' termination options
cma.CMAOptions('verb') # display verbosity options





def cmaes_res(G,M,N):
    G2 = rx.networkx_converter(G)

    def objective_fct(x):
        # x list og node to vaccine
        G3 = G2.copy()
        G3.remove_nodes_from(x.astype(int))
        score = max_ev(G3)
        return score

    x0 = rd.sample(list(G2.node_indices()),M)
    std0 = 10
    #res = cma.CMAEvolutionStrategy(x0, std0).optimize(objective_fct, {'integer_variables': list(range(M)),'bounds': [0,N+1-1e-9]} ).result
    res = cma.fmin2(objective_fct,x0, std0, {'integer_variables': list(range(M)),'bounds': [0,N+1-1e-9]} )
    return res


G = nx.watts_strogatz_graph(1000,10,0.4)
A = nx.adjacency_matrix(G)
eig_init = max_ev(A=A)
M = 300

res = cmaes_res(G,M,1000)


G2 = rx.networkx_converter(G)
G3 = G2.copy()
G3.remove_nodes_from(res[0].astype(int))
score = max_ev(G3)

print(eig_init - score)

