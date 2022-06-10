from cProfile import label
import networkx as nx
import retworkx as rx
import matplotlib.pyplot as plt
import random as rd
from conjugate_gradient import conjugate_gradient_opt,conjugate_gradient_back
import sys
sys.path.append('\graph_immunization')
import network_generation as ng
from vulnerability_meas import max_ev
from netshield import netshield_plus,netshield
import numpy as np
from tqdm import tqdm

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



N = 1000
n_calc=10

'''G = ng.small_world(N)
Gr = rx.networkx_converter(G)
A = rx.adjacency_matrix(Gr)

eig_start = 0
for i in range(n_calc):
    eig_start += max_ev(Gr)

eig_start /= n_calc
# save graph 
#nx.write_gml(G,'graph_benchmark_XXXX.gml')


 ### Solution with conj grad
sol_conj_grad = conjugate_gradient_opt(G,N)

eig_conj = [0]
cost = [0]


l_index = list(range(N))
score1 = 0
for i in tqdm(range(0,N)):
    node = sol_conj_grad[i]
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


### Solution with conj grad back
sol_conj_back = conjugate_gradient_back(G,N)

eig_back = [0]
n_calc=20
A= rx.adjacency_matrix(Gr)
l_index = list(range(N))
score2 = 0
for i in tqdm(range(0,N)):
    node = sol_conj_back[i]
    A = np.delete(A,l_index.index(node),0)
    A = np.delete(A,l_index.index(node),1)
    l_index.remove(node)
    eig=0
    for j in range(n_calc):
        eig += max_ev(A=A)
    eig_back.append(eig_start-eig/n_calc)
    score2+=eig/n_calc
print(score2)

np.save("sol_conj_grad_and_back_smallworld_1006.npy",np.array([cost,sol_conj_grad,sol_conj_back]))

### solution with netshield +
A= rx.adjacency_matrix(Gr)
score3=0
sol_netsh = np.zeros([N-1,N-1])
eg_netsh = [0]
for M in tqdm(range(1,N)):

    vacc = netshield_plus(G,M,int(0.9+0.1*M))
    sol_netsh[M-1,:len(vacc)] = vacc
    vacc = np.array(vacc)
    eig = 0
    for i in range(n_calc):
        eig += f_obj(vacc,A)
    eg_netsh.append(eig_start-eig/n_calc)
    score3+=eig/n_calc
print(score3)
np.save("sol_netshieldp_smallworld_1006.npy",sol_netsh)


'''


### solution with netshield 
G = nx.read_gml('graph_benchmark_1006.gml')
Gr = rx.networkx_converter(G)
A = rx.adjacency_matrix(Gr)

eig_start = 0
for i in range(n_calc):
    eig_start += max_ev(Gr)

eig_start /= n_calc


score4=0
sol_netsh_ = np.zeros([N-1,N-1])
eg_netsh_ = [0]
for M in tqdm(range(1,N)):

    vacc = netshield(Gr,M)
    sol_netsh_[M-1,:len(vacc)] = vacc
    vacc = np.array(vacc)
    eig = 0
    for i in range(n_calc):
        eig += f_obj(vacc,A)
    eg_netsh_.append(eig_start-eig/n_calc)
    score4+=eig/n_calc
print(score4)

np.save("sol_netshield_smallworld_1006.npy",sol_netsh_)





plt.figure(1)
plt.plot(cost,eig_conj,label="conjugate gradient method")
plt.plot(cost,eig_back,label="conjugate gradient back")
plt.plot(cost,eg_netsh,label="netshield_plus")
plt.grid()
plt.xlabel('cost')
plt.ylabel("eigendrop")
plt.legend()
plt.savefig("benchmark_smallworld_1006.png")
plt.show()

#print(score1,score2,score3)

