
include("NetworksUtils.jl")

using .NetworksUtils
using LightGraphs

N = 50

G = small_world(N);

eig = max_ev(G)


print(eig)

