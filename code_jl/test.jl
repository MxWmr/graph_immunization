using LightGraphs


G = LightGraphs.SimpleGraphs.watts_strogatz(1000,6,0.8);

b = betweenness_centrality(G);








