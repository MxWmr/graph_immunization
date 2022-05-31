import numpy as np
import matplotlib.pyplot as plt

mat = np.load('saved_lists\eff_configmodel_all_.npy',allow_pickle=True)
N=1000
ed_cent=mat[1]
ed_deg=mat[0]
ed_netsh=mat[2]
cost = np.arange(0.05,1,0.05)


plt.figure(1)
plt.plot(cost,ed_deg,label="degree max")
plt.plot(cost,ed_cent,label='centrality max')
plt.plot(cost,ed_netsh,label='netshield+')
plt.grid()
plt.legend()
plt.xlabel('Proportion of nodes vaccinated')
plt.ylabel('eigendrop')

plt.show()

