import matplotlib.pyplot as plt
import EoN
import random as rd
from network_generation import config_model, small_world, scale_free
from vulnerability_meas import max_ev
import networkx as nx
from tqdm import tqdm

N=1000
r_spec=[]
r_simu=[]
r_spec2=[]
r_simu2=[]
r_spec3=[]
r_simu3=[]

def simu(G):
    R_last=0
    for j in range(50):
        infected = rd.sample(G.nodes,10)
        t, S, I, R = EoN.fast_SIR(G, 0.6, 1,
                            initial_infecteds = infected)
        R_last+=R[-1]

    return R_last/(S[0]+len(infected))/50

for i in range(200):
    gam=rd.randrange(1,10)*0.51/10
    G = nx.scale_free_graph(N,alpha=0.49,beta=0.51-gam,gamma=gam)
    rs=max_ev(G)
    r_spec.append(rs)
    prop=simu(G)
    r_simu.append(prop)



for i in tqdm(range(100)):
    G = config_model(N,rd.randrange(0,20))
    rs=max_ev(G)
    r_spec2.append(rs)
    prop=simu(G)
    r_simu2.append(prop)


for i in tqdm(range(100)):
    G = G=nx.newman_watts_strogatz_graph(N,rd.randrange(0,20),0.4)
    rs=max_ev(G)
    r_spec3.append(rs)
    prop=simu(G)
    r_simu3.append(prop)


plt.scatter(r_spec,r_simu,label='scale free')
plt.scatter(r_spec2,r_simu2,label='configuration model')
plt.scatter(r_spec3,r_simu3,label='small world')
plt.grid()
plt.legend()
plt.xlabel('rayon spectral')
plt.ylabel('proportion infectée')
plt.show()