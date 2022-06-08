import time 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import EoN
import random as rd
import retworkx as rx


def von_mises(Ao, n, vector =False, eps =0.01, itemax =1000):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector

    A = np.copy(Ao)
    A = np.delete(A,n,0)
    A = np.delete(A,n,1)
    b_k = np.random.rand(A.shape[0])

    b_k1_norm = np.linalg.norm(b_k)
    v=0
    ite=0
    while abs(v-b_k1_norm)>eps and ite<itemax:
        v = b_k1_norm

        # calculate the matrix-by-vector product Ab
        b_k1 = A.dot(b_k)


        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1,ord=2)

        # re normalize the vector

        b_k = 1/(b_k1_norm+0.001)*b_k1

        ite+=1

    r_spec = (b_k.T).dot(A.dot(b_k))/(np.dot(b_k.T,b_k)+0.0001)

    if vector :
        return r_spec,b_k
    else:
        return r_spec


G1 = nx.watts_strogatz_graph(1000,10,0.8)
G= rx.networkx_converter(G1)
A = rx.adjacency_matrix(G)

start = time.time()
r = von_mises(A,10)
stop = time.time()

print(r)
print(stop-start)
