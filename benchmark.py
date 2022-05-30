## Packages
import networkx as nx
import EoN
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from Vaccination_select import centrality_max_no_recomp, deg_max
from network_generation import *
from Vaccination_select import *
from GA import GA
from netshield import netshield_plus


N = 1000    # number of nodes

## Select the graph

#G = small_world(N)
#G = scale_free(N)
G = config_model(N)

## Parameter setting 

n_inf = 10   # initial number of infected nodes

gamma = 1
tau = 0.3

## Vaccination 
num_sim = 1000   #num of simulation for the mean calculus
M_list = range(int(0.05*N),N,int(0.05*N))   

cost=[]
eff=[]

for M in tqdm(M_list):

    ### Vaccination selection

    #vaccinated =centrality_max_recomp(G,M)
    #vaccinated,l_n,l_vuln =GA(G,M, N,  c=2000, mut_r=2, n_gene=2000)
    vaccinated = netshield_plus(G,M,int(M/20))

    G_i=G.copy()
    G_i.remove_nodes_from(vaccinated)


    R_last=0
    for i in range(num_sim):
        
        infected = rd.sample(G_i.nodes,n_inf)

        t, S, I, R = EoN.fast_SIR(G_i, tau, gamma,
                                    initial_infecteds = infected)
        R_last+=R[-1]
    cost.append(M/N)
    eff.append(R_last/N/num_sim)

 

np.save('saved_lists/'+'eff_'+'configmodel_'+'netshield_'+'10inf'+'.npy',eff)

plt.figure(1)
plt.plot(cost,eff)
plt.grid()
plt.title('Genetic optimization for a small world graph')
plt.xlabel('Proportion of nodes vaccinated')
plt.ylabel('Proportion of nodes infected ')
plt.savefig('saved_benchmarks/'+'benchmark_'+'configmodel_'+'netshield_'+'10inf'+'.png')
plt.show()



    