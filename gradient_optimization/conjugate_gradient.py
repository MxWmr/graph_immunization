import numpy as np
import retworkx as rx 


def power_iteration(A,eta):

    return psi,phi


def grad_comput(A,eta):
    psi,phi = power_iteration(A,eta)
    return np.dot(np.diag(phi),np.dot(A.T,psi))







def conjugate_gradient_opt(G,N):
    """
    G: graph to immunize

    Return:
    vaccinated: a list with all node indices ordered by their vaccination 
    """

    G_r = rx.networkx_converter(G)
    vaccinated = []
    A = rx.adjacency_matrix(G_r)
    eta = np.ones([N])

    for n in range(N):

        grad = grad_comput(A,eta)

        node = np.argmax(grad)[0]
        grad[node] = 0

        while eta[node] == 0:
            node = np.argmax(grad)[0]
            grad[node] = 0

        eta[node] = 0

        vaccinated.append(node)


