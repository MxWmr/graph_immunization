from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import retworkx as rx
from vulnerability_meas import max_ev
from Vaccination_select import centrality_max_recomp,deg_max
from GA import GA

N = 1000
graph = 'small_world'
n_calc=20

G = nx.read_gml('graph_benchmark_1006.gml')
Gr = rx.networkx_converter(G)


eig_start = 0
for i in range(n_calc):
    eig_start += max_ev(Gr)

eig_start /= n_calc

#cout = np.arange(0.05,1,0.05)
mat = np.load('sol_conj_grad_and_back_smallworld_1006.npy',allow_pickle=True)
net = np.load('sol_netshield_smallworld_1006.npy')
netp = np.load('sol_netshieldp_smallworld_1006.npy')
cost = mat[0]
conj_grad = mat[1]
conj_back = mat[2]



score=0
cost3 = [0]
eigendrop_GA = [0]
for M in tqdm(range(50,N,20)):
    G_i = Gr.copy()
    vaccinated,l_n,l_vuln = GA(G,M, N, c=8000, mut_r=1.5, n_gene=10000 )
    G_i.remove_nodes_from(vaccinated.astype(int))
    eig=0
    for j in range(n_calc):
        eig += max_ev(G_i)
    score+=eig/n_calc
    eigendrop_GA.append(eig_start-eig/n_calc)
    cost3.append((M+1)/N)
print(score)



np.save('eigd_GA_smallworld_1006_2.npy',np.array(eigendrop_GA))

"""cost3 = [0]
for M in range(50,N,50):
    cost3.append((M+1)/N)

eigendrop_GA = np.load('eigd_GA_smallworld_1006.npy')"""


A = rx.adjacency_matrix(Gr)
eig_conj = [0]
cost = [0]
l_index = list(range(N))
score1 = 0
for i in tqdm(range(0,N)):
    node = conj_grad[i]
    A = np.delete(A,l_index.index(node),0)
    A = np.delete(A,l_index.index(node),1)
    l_index.remove(node)
    eig=0
    for j in range(n_calc):
        eig += max_ev(A=A)
    eig_conj.append(eig_start-eig/n_calc)
    score1+=eig/n_calc
    cost.append((i+1)/N)
print(score1)


eig_back = [0]
A= rx.adjacency_matrix(Gr)
l_index = list(range(N))
score2 = 0
for i in tqdm(range(0,N)):
    node = conj_back[i]
    A = np.delete(A,l_index.index(node),0)
    A = np.delete(A,l_index.index(node),1)
    l_index.remove(node)
    eig=0
    for j in range(n_calc):
        eig += max_ev(A=A)
    eig_back.append(eig_start-eig/n_calc)
    score2+=eig/n_calc
print(score2)




vacc_centr_r=centrality_max_recomp(G,N)
vacc_deg_r=deg_max(G,N)

eigendrop_cent_r = [0]
score = 0
for i in range(1,N+1):
    G_i = Gr.copy()
    G_i.remove_nodes_from(vacc_centr_r[:i])
    eig=0
    for j in range(n_calc):
        eig += max_ev(G_i)
    score+=eig/n_calc
    eigendrop_cent_r.append(eig_start-eig/n_calc)
print(score)

score=0
eigendrop_deg_r = [0]
for i in range(1,N+1):
    G_i = Gr.copy()
    G_i.remove_nodes_from(vacc_deg_r[:i])
    eig=0
    for j in range(n_calc):
        eig += max_ev(G_i)
    score+=eig/n_calc
    eigendrop_deg_r.append(eig_start-eig/n_calc)
print(score)



eig_netp = [0]
A= rx.adjacency_matrix(Gr)
score2 = 0
for i in tqdm(range(0,N-1)):
    node = netp[i][:i].astype(int)
    Ab = np.delete(A,node,0)
    Ab = np.delete(Ab,node,1)
    eig=0
    for j in range(n_calc):
        eig += max_ev(A=Ab)
    eig_netp.append(eig_start-eig/n_calc)
    score2+=eig/n_calc
print(score2)

eig_net = [0]
A= rx.adjacency_matrix(Gr)
score2 = 0
cost2 = [0]
for i in tqdm(range(0,N-1)):
    node = net[i][:i].astype(int)
    Ab = np.delete(A,node,0)
    Ab = np.delete(Ab,node,1)
    eig=0
    for j in range(n_calc):
        eig += max_ev(A=Ab)
    eig_net.append(eig_start-eig/n_calc)
    score2+=eig/n_calc
    cost2.append((i+1)/N)
print(score2)


plt.figure(1)
#plt.plot(cost,eig_conj,label='conjugate gradient')
#plt.plot(cost,eig_back,label='conjugate gradient back')
plt.plot(cost2,eig_net,label='netshield')
plt.plot(cost2,eig_netp,label='netshield+')
plt.plot(cost,eigendrop_deg_r,label="degree centrality ")
plt.plot(cost,eigendrop_cent_r,label="betweenness centrality ")
plt.plot(cost3,eigendrop_GA,label="genetic algorithm")
plt.grid()
plt.legend()
plt.xlabel('Proportion of nodes vaccinated')
plt.ylabel('Eigendrop ')
plt.show()