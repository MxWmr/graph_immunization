## Packages
from cProfile import label
import networkx as nx
import EoN
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from Vaccination_select import centrality_max_recomp, deg_max
from network_generation import *
from vulnerability_meas_nx import max_ev
from GA import GA


N = 1000    # number of nodes

## Select the graph
n_graph = 1
l_G = []
for i in range(n_graph:)
    #G = small_world(N)
    #G = scale_free(N)
    G = config_model(N)
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
ed_ga=[]
ed_cent=[]
ed_deg=[]

for M in tqdm(M_list):

    ### Vaccination selection

    r_=0
    for i_g,G in enumerate(l_G):

        ## GA

        vaccinated,l_n,l_vuln =GA(G,M, N, c=1000, mut_r=2 )

        G_i=G.copy()
        G_i.remove_nodes_from(vaccinated)


        r=0
        for i in range(num_calc):
            r+= max_ev(G_i)

        cost.append(M/N)
        ed_ga.append(rs-r/num_calc)

        ## Degree

        vaccinated = deg_max(G,M)

        G_i=G.copy()
        G_i.remove_nodes_from(vaccinated)


        r=0
        for i in range(num_calc):
            r+= max_ev(G_i)

        ed_deg.append(rs-r/num_calc)


        ## Centrality
        
        vaccinated =centrality_max_recomp(G,M)

        G_i=G.copy()
        G_i.remove_nodes_from(vaccinated)

        r=0
        for i in range(num_calc):
            r+= max_ev(G_i)
        r_+=r/num_calc

    ed_cent.append(l_rs[i_g]-r_/n_graph)



 

#np.save('saved_lists/'+'eff_'+'configmodel_'+'GA2_'+'10inf'+'.npy',eff)

plt.figure(1)
plt.plot(cost,ed_ga,label='genetic algorithm')
plt.plot(cost,ed_deg,label="degree max")
plt.plot(cost,ed_cent,label='centrality max')
plt.grid()
plt.legend()
plt.xlabel('Proportion of nodes vaccinated')
plt.ylabel('eigendrop')
plt.savefig('saved_benchmarks/'+'fullbenchmark_'+'configmodel_'+'eigendrop'+'.png')
plt.show()



    