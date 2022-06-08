
using LightGraphs
using Arpack


function max_ev(Ao,n,vector=false)  
    A = copy(Ao)
    a = A[1:end .!= n, 1:end .!= n]
    val,vect = Arpack.eigs(A,nev=1);
    if vector
        return val,vect
    else
        return val[1]
    end

end 



N = 1000;
G = LightGraphs.SimpleGraphs.watts_strogatz(N,10,0.8);
A = adjacency_matrix(G);

t = @time  r = max_ev(A,10)

