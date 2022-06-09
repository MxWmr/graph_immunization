using LightGraphs, ProgressBars
using LinearAlgebra, Random
using StatsBase
import Arpack


function choose_first_step(known_config,N)
    buffer = 1:N
    proba = zeros(Float64,N)

    for k in 1:N
        try
            proba[k] = known_config[Set([k])][2]
        catch
            proba[k] = 1
        end
    end
    proba /= sum(proba)

    return sample(buffer,Weights(proba))
end

function choose_next_step(config,known_config,N)
    buffer = setdiff(1:N,config)
    proba = zeros(Float64,length(buffer))

    for (k,node) in enumerate(buffer)
        try
            proba[k] = known_config[Set(config+node)][2] ## divide by cost ?
        catch
            proba[k] = 1  #### to change !!!!!
        end
    end
    proba /= sum(proba)

    return sample(buffer,Weights(proba))
end




function f_obj(config, Ao, eps =0.1, itemax =300)
    A = copy(Ao)
    l = Int.(config)
    A = A[setdiff(1:end,l),setdiff(1:end,l)]
    #A = A[Not(l),Not(l)]  ## not so sure of that
    if size(A)[1]>1
        val,vect = Arpack.eigs(A,nev=1);
        return val[1]
    else
        return 0
    end
end



function generate_colony(known_config,μ,N,A)

    colony = zeros(Int,μ,N)
    cost = zeros(Float64,μ)

    for a in 1:μ

        #first step
        node = choose_first_step(known_config,N)
        colony[a,1] = node

        try 
            cost[a]+= known_config[Set([node])][1]
        catch
            local_cost = f_obj([node],A)
            cost[a] += local_cost
            ## create the config
            known_config[Set([node])] = [local_cost,1]
        end

        for ve in 1:N

            node =  choose_next_step(colony[a,1:ve],known_config,N)
            colony[a,ve] = node    

            try
                cost[a]+= known_config[Set(colony[a,1:ve])][1]
            catch
                local_cost = f_obj(colony[a,1:ve],A)
                cost[a] += local_cost
                ## create the config
                known_config[Set(colony[a,1:ve])] = [local_cost,1]
            end
        end
    end

    return colony,cost,known_config
end






function ant_colony(G,N,μ =50,n_gene =1000)
    """
    Return the best order of vaccination of the nodes in G
    in:
        G: graph (LightGraphs)
        N: number of nodes
        mu: number of ant by generation
        n_gene: number of generation max
    out:
        best_ant: nodes ordered by their priority of vaccination
    """

    A = adjacency_matrix(G)

    # repository of knoxn configuration
    known_config = Dict()

    best_cost = 100000

    for n in ProgressBar(1:n_gene)


        # generate new paths and evaluate them
        colony,cost,known_config = generate_colony(known_config,μ,N,A)

        # keep the best ant and its cost
        if minimum(cost) < best_cost
            best_cost = minimum(cost)
            best_ant = colony[argmin(cost),:]
        end

        # Update pheromon
        #known_config = update_pheromon(colony,cost,known_config)    
    
    end


    return best_ant

end



N = 100;
G = LightGraphs.SimpleGraphs.watts_strogatz(N,10,0.8);


vacc = ant_colony(G,N)