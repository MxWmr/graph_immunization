

module Netshield

export netshield,netshield_plus

using LightGraphs
using LinearAlgebra
using .NetworksUtils
using ProgressBars

function  netshield(G,M)
    vaccinated = zeros(Int,M);
    A = adjacency_matrix(G);
    feig,u=max_ev(G,true)
    N = nv(G)
    v = zeros(Int,N);
    score = zeros(Int,N);

    for j in 1:N
        v[j] = (2*feig-A[j,j])*u[j]^2
    end

    for iter in 1:M
        B = A[:,vaccinated]
        b = dot(B,u[vaccinated])

        for j in 1:N
            if j in vaccinated
                score[j] = -1
            else
                score[j] = v[j]-2*b[j]*u[j]
            end
        end
        vaccinate[iter] = argmax(score)
    end
    
end # end netshield


function netshield_plus(G,M,b)
    G2 = squash(G)
    vaccinated = zeros(Int,M);
    t = floor(M/b)

    for j in ProgressBar(1:t)
        vacc_p = netshield(G2,M)
        vaccinated = union(vaccinated,vacc_p)

        for n in vacc_p
            rem_vertex!(G2,v)  ## surtely something to do with node name
        end
    end

    if M > t*b
        vacc_p = netshield(G2,M-t*b)
        vaccinated = union(vaccinated,vacc_p)
    end

    return vaccinated
end # end netshield_plus    



end  # end module