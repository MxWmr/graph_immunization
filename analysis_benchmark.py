from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

N = 1000
graph = 'configmodel'

cout = np.arange(0,1,0.05)

deg = np.load('saved_lists\eff_'+graph+'_degmax_10inf.npy')
centr = np.load('saved_lists\eff_'+graph+'_centralmax_10inf.npy')
ga = np.load('saved_lists\eff_'+graph+'_GA2_10inf.npy')


plt.figure(1)
plt.plot(cout,deg,label='degree max')
plt.plot(cout,centr,label='centrality max')
cout = np.arange(0.05,1,0.05)
plt.plot(cout,ga,label='genetic algorithm')
plt.grid()
plt.legend()
plt.xlabel('Proportion of nodes vaccinated')
plt.ylabel('Proportion of nodes infected ')
plt.show()