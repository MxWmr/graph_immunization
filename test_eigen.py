import numpy as np
import networkx as nx
import retworkx as rx
from vulnerability_meas import max_ev
from network_generation import *


N = 1000

G = small_world(N)
G2 = rx.networkx_converter(G)
A = rx.adjacency_matrix(G2)

w,vect = np.linalg.eig(A)
idfeig = np.argmax(np.absolute(w))
feig = w[idfeig]
u = vect[idfeig]




feig2,u2=max_ev(A=A,vector=True)

print(u2-u)