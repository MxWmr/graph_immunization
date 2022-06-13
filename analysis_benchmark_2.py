from json import load
import numpy as np
import matplotlib.pyplot as plt

N = 1000
graph = 'small_world'

cout = np.arange(0.05,1,0.05)
mat = np.load('saved_lists\ed_conj_grad_smallworld.npy')
mat2 = np.load('saved_lists\ed_smallworld_all3_.npy')
mat3 = np.load('saved_lists\eff_smallworld_cma_.npy')
deg = mat2[0]
centr = mat2[1]
ga = mat2[3]
ns = mat2[2]
cout2 = mat[0]
conjg = mat[1]
rand = mat[2]
cost3 = mat3[0]
cma = mat3[1]


plt.figure(1)
plt.plot(cout,deg,label='degree max')
plt.plot(cout,centr,label='centrality max')
plt.plot(cout,ga,label='genetic algorithm')
plt.plot(cout,ns,label='netshield+')
plt.plot(cout2,conjg,label='conjugate gradient')
plt.plot(cout2,rand,label='random vacc')
plt.plot(cost3,cma,label='CMA-ES')
plt.grid()
plt.legend()
plt.xlabel('Proportion of nodes vaccinated')
plt.ylabel('Eigendrop ')
plt.show()