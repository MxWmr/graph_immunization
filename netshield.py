import retworkx as rx
import networkx as nx
import numpy as np
from tqdm import tqdm



def netshield(G,M):

    G2 = rx.networkx_converter(G)
    vaccinated=[]
    A = rx.adjacency_matrix(G2)


    w,vect = np.linalg.eig(A)
    idfeig = np.argmax(np.absolute(w))
    feig = w[idfeig]
    u = vect[idfeig]

    v=np.zeros([len(u)])
    score=np.zeros([len(u)])

    for j in range(len(u)):
        v[j]=(2*feig-A[j,j])*u[j]**2
    
    for iter in range(M):
        B = A[:,vaccinated]
        b = np.dot(B,u[vaccinated])

        for j in range(len(u)):
            if j in vaccinated:
                score[j] = -1
            else:                
                score[j] = v[j]-2*b[j]*u[j]
        
        vaccinated.append(np.argmax(score))

    return vaccinated


def netshield_plus(G,M,b):

    G2 = rx.networkx_converter(G)
    vaccinated=[]
    A = rx.adjacency_matrix(G2)
    t=int(np.floor(M/b))

    for j in tqdm(range(t)):
        vacc_p = netshield(G,M)
        vaccinated = list( set(vaccinated).union(vacc_p))

        G2.remove_nodes_from(vacc_p)
        A = rx.adjacency_matrix(G2)
    
    if M > t*b :
        vacc_p = netshield(A,M-t*b)
        vaccinated = list( set(vaccinated).union(vacc_p))
    
    return vaccinated




#from network_generation import *

#N = 1000    # number of nodes

#G = small_world(N)
#M = int(0.3*N)
#vaccinated = netshield_plus(G,M,10)

