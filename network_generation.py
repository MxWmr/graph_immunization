import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 


# Configuration model with a gaussian distribution of degree
def config_model(N, med =10):
    std= 3 # standard deviation 
    distrib = np.random.normal(med,std,N).astype(int)
    if np.sum(distrib)%2 != 0:
        distrib[0]+=1
    G = nx.configuration_model(distrib)
    return G


# scale free graph with power law distribution 
def scale_free(N):
    G = nx.scale_free_graph(N)
    return G

def small_world(N):
    G=nx.newman_watts_strogatz_graph(N,10,0.4)
    return G
