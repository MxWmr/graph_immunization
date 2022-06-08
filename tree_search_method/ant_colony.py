import numpy as np
import retworkx as rx
import random as rd
from tqdm import tqdm

def choose_first_step(known_config,N):
    buffer = range(N)
    proba = np.zeros([N])

    for k in range(N):
        
        if frozenset([k]) in list(known_config.keys()):
            proba[k] = known_config[frozenset([k])][1] ## divide by cost ?
        else:
            if k == 99:
                proba[99] = 1  #### to change !!!!!
            else:
                proba[k] = 1  #### to change !!!!!
    proba /= np.sum(proba)

    return np.random.choice(buffer,p=proba)    



def choose_next_step(config,known_config,N):
    buffer = [i for i in range(N) if i not in config]
    proba = np.zeros([len(buffer)])
    for k,node in enumerate(buffer):
        if frozenset(config+node) in list(known_config.keys()):
            proba[k] = known_config[frozenset(config+node)][1] ## divide by cost ?
        else:
            proba[k] = 1  #### to change !!!!!
    proba /= np.sum(proba)

    return np.random.choice(buffer,p=proba)
            


def f_obj(config, Ao, eps =0.1, itemax =100):

    A = np.copy(Ao)
    l = config.astype(int)
    A = np.delete(A,l,0)
    A = np.delete(A,l,1)


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

    return r_spec



def generate_colony(known_config,mu,N,A):

    colony = np.empty([mu,N])
    cost = np.zeros([mu])

    for a in range(mu):     #for each ant

        # first step 
        node = choose_first_step(known_config,N)
        colony[a,0] = node

        if frozenset([node]) in list(known_config.keys()):
            cost[a]+= known_config[frozenset([node])][0]
        else:
            local_cost = f_obj(np.array([node]),A)
            cost[a] += local_cost
            ## create the config
            known_config[frozenset([node])] = [local_cost,1]

        for ve in range(1,N):  #for each vertex of the path

            node =  choose_next_step(colony[a,:ve],known_config,N)

            colony[a,ve] = node

            if frozenset(colony[a,:(ve+1)]) in list(known_config.keys()):
                cost[a]+= known_config[frozenset(colony[a,:(ve+1)])][0]
            else:
                local_cost = f_obj(colony[a,:(ve+1)],A)
                cost[a] += local_cost
                ## create the config
                known_config[frozenset(colony[a,:(ve+1)])] = [local_cost,1]


    return colony,cost





def update_pheromon(colony,cost,known_config,Q = 100,p = 0.2):

    #evaporate the pheromons
    for config in list(known_config.keys()):
        known_config[config][1] *= (1-p)

    # add new the pheromons
    for i,a in enumerate(colony):
        for j in range(N):
            known_config[frozenset(a[:j])][1] +=Q/cost[i]
    
    
            



def ant_colony(G,N,mu =50, n_gene = 1000):
    """
    Return the best order of vaccination of the nodes in G
    in:
        G: graph (networkx)
        N: number of nodes
        mu: number of ant by generation
        n_gene: number of generation max
    out:
        best_ant: nodes ordered by their priority of vaccination
    """

    Gr = rx.networkx_converter(G)
    A = rx.adjacency_matrix(Gr)

    ### Repository of known configuration
    known_config = dict()

    best_cost =np.Inf

    for n in tqdm(range(n_gene)):

        # generate new paths and evaluate them
        colony,cost = generate_colony(known_config,mu,N,A)
        

        # keep the best ant and itd cost
        if np.amin(cost) < best_cost:
            best_cost = np.amin(cost)
            best_ant = colony[np.argmin(cost),:]


        # Update pheromon
        known_config = update_pheromon(colony,cost,known_config)


    return best_ant



import networkx as nx
N = 100

G = nx.watts_strogatz_graph(N,10,0.8)


vacc = ant_colony(G,N)