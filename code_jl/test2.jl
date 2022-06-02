using LightGraphs


G = SimpleGraph(3);

add_edge!(G, 1, 2);

add_edge!(G, 1, 3);



rem_vertex!(G, 1);

h = SimpleGraphFromIterator(edges(G));

print(collect(edges(h)))




