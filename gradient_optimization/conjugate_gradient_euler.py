import numpy as np
import retworkx as rx 
from tqdm import tqdm
import sys
sys.path.append('\graph_immunization')
from vulnerability_meas import max_ev


def grad_comput(A,l_index):
    fe = max_ev(A=A)
    #B = np.dot(np.diag(eta),np.dot(A,np.diag(eta)))
    grad = np.zeros_like(l_index)
    for i in range(len(l_index)):
        Ab = np.delete(A,i,0)
        Ab = np.delete(Ab,i,1)
        grad[i] = fe - max_ev(A=Ab)
    return grad


def conjugate_gradient_euler(G,N,exact=False):
    """
    in:
    G: graph to immunize
    N: number of nodes

    out:
    vaccinated: a list with all node indices ordered by their vaccination 
    """

    G_r = rx.networkx_converter(G)
    vaccinated = []
    A = rx.adjacency_matrix(G_r)

    l_index = list(range(N))
    for n in tqdm(range(N)):

        grad = grad_comput(A,l_index)

        try:
            node = np.argmax(grad)[0]
        except:
            node = np.argmax(grad)

        A = np.delete(A,node,0)
        A = np.delete(A,node,1)

        node = l_index[node]

        l_index.remove(node)
        vaccinated.append(node)

    return vaccinated




'''import networkx as nx
N = 1000

G = nx.watts_strogatz_graph(N,10,0.8)

vacc = conjugate_gradient_opt(G,N)
print(vacc)'''