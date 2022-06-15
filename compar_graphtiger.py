import retworkx as rx
import networkx as nx
from netshield import netshield
from graph_tiger.attacks import get_node_ns

G = nx.read_gml('graph_benchmark_1006.gml')
Gr = rx.networkx_converter(G)
M = 10
vacc1 = netshield(Gr,M)