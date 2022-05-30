#import matplotlib
#matplotlib.use('TKAgg')

from turtle import color
import matplotlib.pyplot as plt 
import numpy as np
import networkx as nx
import retworkx as rx
import random as rd
from vulnerability_meas import *
from tqdm import tqdm 
import time 
plt.style.use('ggplot')
import EoN

def live_plot(l_vuln,line1,n):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot([0],l_vuln,'-o',alpha=0.8)        
        #update plot label/title
        plt.grid()
        plt.xlabel('generation')
        plt.ylabel('vulnerability')
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_data(range(n),l_vuln)

    # adjust limits if new data goes beyond bounds
    if np.min(l_vuln)<=line1.axes.get_ylim()[0] or np.max(l_vuln)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(l_vuln)-np.std(l_vuln),np.max(l_vuln)+np.std(l_vuln)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.00001)
    
    # return line so we can update it again in the next iteration
    return line1


def calc_vuln(G,popu, measure, vuln_elite):
    vulnerability = np.zeros([len(popu)])
    start = 0
    if vuln_elite is not None:
        for j in range(len(vuln_elite)):
            vulnerability[j] = vuln_elite[j]
        start = len(vuln_elite)

    if measure == 'max_ev':

        for i in range(start,len(popu)): 
            G_i=G.copy()
            G_i.remove_nodes_from(popu[i,:].astype(int)) #0.2 s per iter 
            vulnerability[i] = max_ev(G_i) # 0.2 s per iter        


    elif measure == 'perco_estim':  # odd

        for i in range(start,len(popu)):  
            G_i=G.copy()
            G_i.remove_nodes_from(popu[i,:].astype(int)) #0.2 s per iter 
            vulnerability[i] = perco_estim(G_i, 10)

    elif measure == 'simu_mes': # long...

        for i in range(start,len(popu)):
            G_i=G.copy()
            G_i.remove_nodes_from(popu[i,:].astype(int)) #0.2 s per iter 
            vulnerability[i] = simu_mes(G_i, n_sim=5)

    elif measure == 'mean_field':   # not build

        for i in range(start,len(popu)):
            G_i=G.copy()
            G_i.remove_nodes_from(popu[i,:].astype(int)) #0.2 s per iter 
            vulnerability[i] = mean_field(G_i)

    else:
        raise ValueError('Invalid measure')

    return vulnerability



def parents_selection(vulnerability, n_select, L):
    ids=[]
    buffer_id=list(range(L))+list(range(L))

    ## Unbiased tournament selection
    for k in range(n_select):
        [id1,id2]=np.random.choice(buffer_id,2,replace=False)

        while id1 == id2:
            [id1,id2]=np.random.choice(buffer_id,2,replace=False)

        if vulnerability[id1]<vulnerability[id2]:
            ids.append(id1)
        else:
            ids.append(id2)

        buffer_id.remove(id1)
        buffer_id.remove(id2)


    ## avoiding two parents consecutive
    prev=ids[0]
    k=1
    while k<len(ids):
        if ids[k]==prev and k!=len(ids)-1:
            del ids[k]
            ids.append(prev)
            k-=1
        elif ids[k]==prev:
            del ids[k]
            ids.insert(0,prev)
        else:
            prev=ids[k]
        k+=1 
    

    return ids 



def crossover(parents, L, n_select ,M):
    offsprings=np.empty([L-n_select,len(parents[0])])

    # prepare list of parents for crossover
    buffer_par = list(range(n_select))
    
    # append parents if it's necessary
    if n_select < L/2:
        buffer_par.extend(buffer_par[0:L/2-n_select])
    
    # randomize the crossover
    #rd.shuffle(buffer_par)

    for i in range((L-n_select)//2):  # L-n_select must be even

        # choose randomly the point of crossover
        cross_point = rd.randint(1,M)

        cross1 = rd.sample(list(parents[buffer_par[2*i],:]),cross_point)
        offsprings[2*i,0:cross_point] = cross1


        cross2 = rd.sample(list(parents[buffer_par[2*i+1],:]),cross_point)
        offsprings[2*i+1,0:cross_point] = cross2


        cross3 = list(set(parents[buffer_par[2*i+1],:]).difference(cross1).difference(cross2))
        if len(cross3) < M-cross_point:
            cross3.extend(rd.sample(list(set(parents[buffer_par[2*i+1],:]).difference(cross1).difference(cross3)),M-cross_point-len(cross3)))
        else:
            cross3=  rd.sample(cross3,M-cross_point)
        
        offsprings[2*i,cross_point:M] = cross3


        cross4 = list(set(parents[buffer_par[2*i],:]).difference(cross2).difference(cross1))
        if len(cross4) < M-cross_point:
            cross4.extend(rd.sample(list(set(parents[buffer_par[2*i],:]).difference(cross2).difference(cross4)),M-cross_point-len(cross4)))
        else:
            cross4=  rd.sample(cross4,M-cross_point)

        offsprings[2*i+1,cross_point:M] = cross4

    return offsprings


def mutation(offsprings, mut_r, M, N):

    for i in range(len(offsprings)):
        for j in range(M):
            if np.random.binomial(1,mut_r*1/M):
                offsprings[i,j] = rd.choice(list(set(range(N)).difference(offsprings[i,:])))

    return offsprings    


def GA(G, M, N, L=52, f_eps =0.0001, n_gene=1000, n_select =None, measure  ='max_ev', mut_r =1 , c =50 , verbose =0):
    G_r=rx.networkx_converter(G)
    '''
    G: the graph to immunize
    M: the number of node to vaccinate
    L: the number of individuals by generation
    n_gene: number of generation  -> to change with a epsilon ? 
    measure: method for measuring the vulnerability: 'max_ev' 'perco_estim' 'simu_mes' 'mean_field'
    n_select: number of individual to select each generation, max=L/2     
        L-n_select must be even !!!
    mut_r: mutation rate, 1 for normal mutation, 0 for no mutation, n for n times more 
    '''
    if n_select is None:
        n_select = int(L/2)

    ## create the initial population
    popu=np.empty([L,M])
    for i in range(L):
        popu[i,:] =  rd.sample(list(G_r.node_indices()),M)
    
    l_vuln=[]
    l_n=[]
    l_n2=[]
    l_simu=[]
    
    if verbose>0:
        line1 = []
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
       
        # create a variable for the line so we can later update it
        line1, = ax.plot([],[],'r-o',alpha=0.8,label='vuln')  

        if verbose>1:
            ax2 = ax.twinx() 
            line2, = ax2.plot([],[],'b-*',alpha=0.8,label='simu')        
            ax2.set_ylabel('proportion of infected')
            ax2.legend()

        plt.xlabel('generation')
        ax.legend()
        ax.set_ylabel('vulnerability')
        plt.show()


    old_vuln = 0
    incr = 0
    vuln_elite= None
    for n in tqdm(range(n_gene)):

        #start = time.time()
        ## computation of the vulnerability  -> it take the time !
        vulnerability = calc_vuln(G_r,popu, measure, vuln_elite)
        #stop = time.time()
        #print(stop-start)
        new_vuln = min(vulnerability)

        ## Selecting the best parents for the new generation
        id_parents = parents_selection(vulnerability, n_select, L)
        parents = popu[id_parents,:]


        ## Keep the vulnerabilty of the elite
        id_elite = np.argsort(vulnerability)[:n_select]
        vuln_elite = np.take(vulnerability, id_elite)


        ## Generating new generation with crossover
        offsprings = crossover(parents, L, n_select,M)
        

        ## Adding mutations
        offsprings = mutation(offsprings, mut_r,M,N)

        
        ## Creating the new generation: elite and offspring
        popu[0:n_select,:] = popu[id_elite,:]
        popu[n_select:L,:] = offsprings


        ## Keep some data 
        if n%10 ==0 :
            l_vuln.append(new_vuln)
            l_n.append(n)
            
            if  verbose>0:
                line1.set_data(l_n,l_vuln)

                # adjust limits if new data goes beyond bounds
                if np.min(l_vuln)<=line1.axes.get_ylim()[0] or np.max(l_vuln)>=line1.axes.get_ylim()[1]:
                    ax.set_ylim([np.min(l_vuln)-np.std(l_vuln),np.max(l_vuln)+np.std(l_vuln)])
                if l_n[-1]>=line1.axes.get_xlim()[1]:
                    ax.set_xlim([0,l_n[-1]+50])

                if n%50 == 0 and verbose>1:
                    R_last = 0
                    l_n2.append(n)
                    for h in range(100):
                        G_i= G.copy()
                        G_i.remove_nodes_from(popu[0,:])
                        infected = rd.sample(G_i.nodes,10)
                        t, S, I, R = EoN.fast_SIR(G_i, 0.3, 1,
                                                    initial_infecteds = infected)
                        R_last+=R[-1]

                    l_simu.append(R_last/N/100)
                    
                    line2.set_data(l_n2,l_simu)

                    # adjust limits if new data goes beyond bounds
                    if np.min(l_simu)<=line2.axes.get_ylim()[0] or np.max(l_simu)>=line2.axes.get_ylim()[1]:
                        ax2.set_ylim([np.min(l_simu)-np.std(l_simu),np.max(l_simu)+np.std(l_simu)])
                    if l_n2[-1]>=line2.axes.get_xlim()[1]:
                         ax2.set_xlim([0,l_n2[-1]+50])

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
        
        

        ## Stop the algo with a condition
        if abs(new_vuln-old_vuln)<f_eps*old_vuln:
            incr+=1
        else:
            incr=0

        old_vuln = new_vuln

        if incr == c:
            break

        

    print(new_vuln)
    if verbose == 3:
        plt.savefig('ga_evolution.png')

    return popu[0,:],l_n,l_vuln




#from network_generation import *

#N = 1000    # number of nodes#

#G = small_world(N)
#G = scale_free(N)
#G = config_model(N)

#psi = 0.3   #proportion of node vaccinated
#M = int(N*psi)   #Number of vaccinated node


#vaccinated,l_n,l_vuln =GA(G,M,N, verbose=3, c=2000, mut_r=2, n_gene=2000 )

