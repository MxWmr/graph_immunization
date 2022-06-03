

module NetworksUtils



using LightGraphs
using Random,Distributions
Random.seed!(123);
import Arpack
using ProgressBars

## Network_generation
export config_model,power_law,small_world

function  config_model(N, med =10, std = 3)
    k = rand(Normal(med,std),N);
    k = round.(Int, k);
    if sum(k)%2 != 0
        k[1]+=1;    
    end
    G = LightGraphs.SimpleGraphs.random_configuration_model(N,k);

    return G
end

function power_law(N)
    G = LightGraphs.SimpleGraphs.static_scale_free(N,3*N,2.5);
    return G
end

function small_world(N)
    G = LightGraphs.SimpleGraphs.watts_strogatz(N,10,0.4);
    return G
end




## Vaccination selection
export deg_max,centrality_max_recomp 

function deg_max(G,M)
    deg = degree_centrality(G,normalize=false);
    sort = sortperm(deg);
    return sort[end-M+1:end]
end

function centrality_max_no_recomp(G,M)
    cent =  betweenness_centrality(G);
    sort = sortperm(cent);
    return sort[end-M+1:end]
end

function centrality_max_recomp(G,M)
    G2 = squash(G)
    vaccinated = zeros(Int,M);
    for i in ProgressBar(1:M)
    #for i in 1:M
        cent =  betweenness_centrality(G2,normalize=false);
        node = argmax(cent);
        sort_vaccinated = sort(vaccinated)
        for j in sort_vaccinated
            if node>= j && j!=0
                node+=1;
            end
        end
        vaccinated[i] = node;
        
        
    end
    return vaccinated
end


## Vulnerability measure
export max_ev

function max_ev(G,vector=false)
    A = adjacency_matrix(G);
    val,vect = Arpack.eigs(A,nev=1);
    if vector
        return val,vect
    else
        return val[1]
    end

end 

N = 1000;
G = small_world(N);
ψ = 0.3;
M = Int(N*ψ);
vaccinated = centrality_max_recomp(G,M);
G_i = squash(G)

for i in 1:length(vaccinated)
    node = vaccinated[i]
    rem_vertex!(G_i,node)
    for j in (i+1):length(vaccinated)
        if vaccinated[j] > node
            vaccinated[j]-=1
        end
    end
end

println(max_ev(G)-max_ev(G_i))

vaccinated = deg_max(G,M);
G_i = squash(G)

for i in 1:length(vaccinated)
    node = vaccinated[i]
    rem_vertex!(G_i,node)
    for j in (i+1):length(vaccinated)
        if vaccinated[j] > node
            vaccinated[j]-=1
        end
    end
end

println(max_ev(G)-max_ev(G_i))

vaccinated = centrality_max_no_recomp(G,M);
G_i = squash(G)

for i in 1:length(vaccinated)
    node = vaccinated[i]
    rem_vertex!(G_i,node)
    for j in (i+1):length(vaccinated)
        if vaccinated[j] > node
            vaccinated[j]-=1
        end
    end
end

println(max_ev(G)-max_ev(G_i))
end #end of module