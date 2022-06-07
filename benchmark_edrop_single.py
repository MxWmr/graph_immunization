## Packages
from cProfile import label
import networkx as nx
import EoN
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from cmaes_with_module import cmaes_res
from network_generation import *
from vulnerability_meas_nx import max_ev


N = 1000    # number of nodes

## Select the graph
n_graph = 3
l_G = []
for i in range(n_graph):
    G = small_world(N)
    #G = scale_free(N)
    #G = config_model(N)
    l_G.append(G)

## Spectral radius
num_calc = 10   #num of calculuse for approximating the spectral radius
l_rs=[]
for G in l_G:
    rs=0
    for i in range(num_calc):
        rs+= max_ev(G)

    rs/=num_calc
    l_rs.append(rs)

## Vaccination 

M_list = range(int(0.05*N),N,int(0.05*N))   

cost=[]
ed_cma=[]


for M in tqdm(M_list):
    cost.append(M/N)
    ### Vaccination selection

    rcma=0

    for i_g,G in enumerate(l_G):

        ## GA

        vaccinated = cmaes_res(G,M,N)

        G_i=G.copy()
        G_i.remove_nodes_from(vaccinated.astype(int))


        r=0
        for i in range(num_calc):
            r+= max_ev(G_i)
        
        rcma+=r/num_calc

    ed_cma.append(l_rs[i_g]-rcma/n_graph)





mat = np.array([cost,ed_cma])

np.save('saved_lists/'+'eff_'+'smallworld_'+'cma_'+'.npy',mat)

plt.figure(1)
plt.plot(cost,ed_cma,label='genetic algorithm')
plt.grid()
plt.legend()
plt.xlabel('Proportion of nodes vaccinated')
plt.ylabel('eigendrop')
plt.savefig('saved_benchmarks/'+'benchmark_cma'+'smallworld_'+'eigendrop3'+'.png')
plt.show()



    