


module GeneticAlgorithm
export genetic_algorithm

using LightGraphs
using .NetworksUtils
import StatsBase
using ProgressBars
using Random
using Distributions


function calc_vuln(G,L,popu,vuln_elite)
    vulnerability = zeros([L])
    start = 0
    if vuln_elite !== nothing
        start = length(vuln_elite)
        vulnerability[1:start] = vuln_elite
    end
    
    for i in 1:L
        G_i = squash(G)
        rem_vertex!(G_i,popu[i,:])
        vulnerability[i] = max_ev(G_i)
    end

    return vulnerability

end ## end calc_vuln

function parents_selection(vulnerability,n_select,L)
    ids = [];
    buffer_id = vcat(1:L,1:L);

    ## Unbiased tournament selection
    for k in 1:n_select
        id1,id2 = StatsBase.sample!(buffer_id,2,replace=false)
        while id1 == id2
            id1,id2 = StatsBase.sample!(buffer_id,2,replace=false)
        end
        if vulnerability[id1]<vulnerability[id2]
            append!(ids,id1)
        else
            append!(ids,id2)
        end
        pop!(buffer_id,id1)
        pop!(buffer_id,id2)
    end 

    ## avoiding two parents consecutive
    prev=ids[1]
    k=1
    while k<length(ids)
        if ids[k]==prev && k!=length(ids)-1
            deleteat!(ids,k)
            append!(ids,prev)
            k-=1
        elseif ids[k]==prev
            deleteat!(ids,k)
            insert!(ids,0,prev)
        else
            prev = ids[k]
        end
        k+=1
    end

    return ids

end ## end parents_selection

function  crossover(parents, L, n_select ,M)
    offsprings = zeros(Int,[L-n_select,length(parent[0])])

    # prepare listf of parents for crossover
    buffer_par = 1:n_select

    
    for i in 1:((L-n_select)//2)

        # choose randomly the point of crossover
        cross_point = rand(Int,1:M)


        cross1 = StatsBase.sample!(parents[buffer_par[2*i],:],cross_point,replace=false)
        offsprings[2*i,1:cross_point] = cross1


        cross2 =  StatsBase.sample!(parents[buffer_par[2*i+1],:],cross_point,replace=false)
        offsprings[2*i+1,1:cross_point] = cross2


        cross3 = setdiff(setdiff(parents[buffer_par[2*i+1],:],cross1),cross2)
        if length(cross3) < M-cross_point
            append!(cross3, StatsBase.sample!(setdiff(setdiff(parents[buffer_par[2*i+1],:],cross1),cross3),M-cross_point-length(cross3),replace=false))
        else
            cross3 = StatsBase.sample!(cross3,M-cross_point)
        end
        offsprings[2*i,cross_point+1:M] = cross3


        cross4 = setdiff(setdiff(parents[buffer_par[2*i],:],cross2),cross1)
        if length(cross4) < M-cross_point
            append!(cross4, StatsBase.sample!(setdiff(setdiff(parents[buffer_par[2*i],:],cross2),cross4),M-cross_point-length(cross4),replace=false))
        else
            cross4 = StatsBase.sample!(cross4,M-cross_point)
        end
        offsprings[2*i+1,cross_point+1:M] = cross4

    end

return offsprings 

end # end of crossover


function mutation(offsprings,mut_r,M,N)

    d = Bernoulli(mut_r*1/M)
    for i in 1:length(offsprings)
        for j in 1:M
            if rand(d,1)
                offsprings[i,j] = StatsBase.sample(setdiff(1:N,offsprings[i,:]))
            end
        end
    end
           
    return offsprings
    
end #end of mutatopn



function genetic_algorithm(G, M, N, L=52, n_gene=1000, n_select =nothing, mut_r =1)
    """
    G: the graph to immunize
    M: the number of node to vaccinate
    L: the number of individuals by generation
    n_gene: number of generation  -> to change with a epsilon ? 
    n_select: number of individual to select each generation, max=L/2     
        L-n_select must be even !!!
    mut_r: mutation rate, 1 for normal mutation, 0 for no mutation, n for n times more 
    """

    if n_select === nothing
        n_select = Int(L/2)
    end

    ## create initial population

    popu = zeros(Int,[L,M])
    for i in 1:L
        popu[i,:] = StatsBase.sample!(1:N,M,replace=false)
    end

    old_vuln = 0
    incr = 0
    vuln_elite = nothing

    for n in ProgressBar(1:n_gene)

        ## computation of the vulnerability
        Vulnerability = calc_vul(G,L,popu,vuln_elite)

        ## Selecting the best parents for the new generation
        id_parents = parents_selection(vulnerability, n_select, L)
        parents = popu[id_parents,:]


        ## Keep the vulnerabilty of the elite
        id_elite = sortperm(Vulnerability,rev = true)[:n_select]
        vuln_elite = Vulnerability[id_elite]
        # vuln_elite = getindex(vulnerability,id_elite)

        ## Generating new generation with crossover
        offsprings = crossover(parents, L, n_select,M)

        ## Adaptative mutation rate
        mut_r_adapt = mut_r + mut_r*n/1000

        ## Adding mutations
        offsprings = mutation(offsprings, mut_r_adapt,M,N)

        ## Creating the new generation: elite and offspring
        popu[0:n_select,:] = popu[id_elite,:]
        popu[n_select:L,:] = offsprings

    end ## end for 



end # end of genetic_algorithm




end # end of module