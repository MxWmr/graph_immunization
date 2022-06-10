from cmath import inf
import numpy as np
import retworkx as rx
import random as rd
from tqdm import tqdm
import time


def power_iteration(B,eps =0.01, itemax =1000,N=100):
    

    b_k = np.random.rand(B.shape[0])

    b_k1_norm = np.linalg.norm(b_k)
    v=0
    ite=0
    while abs(v-b_k1_norm)>eps and ite<itemax:
        v = b_k1_norm

        # calculate the matrix-by-vector product Ab
        b_k1 = B.dot(b_k)


        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1,ord=2)

        # re normalize the vector

        b_k = 1/(b_k1_norm+0.001)*b_k1

        ite+=1

    phi = np.where(b_k!=0.,1/b_k/N,0)

    return b_k,phi


def grad_comput(A,config,buffer):
    eta = np.ones([A.shape[0]])
    eta[config.astype(int)] = 0
    B = np.dot(A,np.diag(eta))
    psi,phi = power_iteration(B)
    grad = np.dot(np.diag(phi),np.dot(A.T,psi))
    grad = grad[buffer]

    #print('grad',grad)
    return grad



def choose_first_step(known_config,N,alpha,beta):
    buffer = range(N)
    proba = np.zeros([N])
    for k in range(N):    
        try:
            proba[k] = known_config[frozenset([k])][1]**alpha/known_config[frozenset([k])][0]**beta
        except:
            proba[k] = 0.01 #### to change !!!!!
         
    proba /= np.sum(proba)
    return np.random.choice(buffer,p=proba)    


def choose_next_step(A,config,known_config,N,alpha,beta):
    buffer = [i for i in range(N) if i not in config]
    grad  = grad_comput(A,config,buffer)
    proba = np.zeros([len(buffer)])
    for k,node in enumerate(buffer):
        try:
            proba[k] = known_config[frozenset(np.append(config,node))][1]**alpha-grad[k]**beta
        except:
            proba[k] = -grad[k]**beta #### to change !!!!!
    
    proba[proba == -np.inf] = -1e100
    #print("amin",np.amin(proba))
    proba -= np.amin(proba)  
    if np.sum(proba)!=0:
        proba /= np.amax(proba)
        proba /= np.sum(proba)
        try:
            return np.random.choice(buffer,p=proba) 
        except:
            print(proba)
    else:
        return np.random.choice(buffer) 
            


def f_obj(config, Ao, eps =0.1, itemax =300):

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



def generate_colony(known_config,mu,N,A,alpha,beta):

    colony = np.empty([mu,N])
    cost = np.zeros([mu])

    for a in range(mu):     #for each ant

        # first step 
        node = choose_first_step(known_config,N,alpha,beta)
        colony[a,0] = node

        try:
            cost[a]+= known_config[frozenset([node])][0]
        except:
            local_cost = f_obj(np.array([node]),A)
            cost[a] += local_cost
            ## create the config
            known_config[frozenset([node])] = [local_cost,0.1]


        for ve in range(1,N-1):  #for each vertex of the path
            node =  choose_next_step(A,colony[a,:ve],known_config,N,alpha,beta)
            colony[a,ve] = node
            try:
                cost[a]+= known_config[frozenset(colony[a,:(ve+1)])][0]
            except:
                local_cost = f_obj(colony[a,:(ve+1)],A)
                cost[a] += local_cost
                ## create the config
                known_config[frozenset(colony[a,:(ve+1)])] = [local_cost,0.1]

        # choose the last node 
        node =  [i for i in range(N) if i not in colony[a,:(N-1)]][0]
        colony[a,N-1] = node

    return colony,cost,known_config





def update_pheromon(colony,cost,known_config,Q,p,mu):

    #evaporate the pheromons
    for config in list(known_config.keys()):
        known_config[config][1] *= (1-p)



    #select the best ants   
    chosen_ants = colony[np.argsort(cost)[-mu//2:],:]  # here we take the better half of the colony

    # add new the pheromons
    for i,a in enumerate(chosen_ants):
        
        for j in range(N-1):
            known_config[frozenset(a[:(j+1)])][1] +=Q/cost[i]
    
    return known_config
    
    
            



def ant_colony(G,N,mu =50, n_gene = 1000, alpha=0.8, beta=1.01, Q=500, p=0.05):
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
    n_k_old = 0

    for n in tqdm(range(n_gene), position=0, leave=True):

        
        # generate new paths and evaluate them
        colony,cost,known_config = generate_colony(known_config,mu,N,A,alpha,beta)

        # keep the best ant and its cost
        if np.amin(cost) < best_cost:
            best_cost = np.amin(cost)
            best_ant = colony[np.argmin(cost),:]

        print(best_cost)
        # Update pheromon
        known_config = update_pheromon(colony,cost,known_config,Q,p,mu)

        n_k =len(known_config)
        print(n_k-n_k_old)
        n_k_old = n_k
        #if n%5 ==0:
        #    np.save('ant_colony.npy',best_ant)

    return best_ant



import networkx as nx
N = 100

G = nx.watts_strogatz_graph(N,10,0.8)


vacc = ant_colony(G,N)