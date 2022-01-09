import pandas as pd
from tqdm import tqdm
import re

from collections import defaultdict
import networkx as nx
from miRNA_databases import get_HMDD_disease_associated_db

class Graph:
    def __init__(self, vertices):
        self.V = vertices

        self.graph = defaultdict(list)

        self.score = 0

        self.visited = [False]*(self.V)

        self.edges = []
    
    def addEdge(self, u, v):
        self.graph[u].append(v)
    
    def getInfluence(self, S, shortest_path_dict):
        try:
            nodes_in_network =  nx.single_source_shortest_path(nx.DiGraph(self.graph), S).keys() #https://codegolf.stackexchange.com/questions/26031/find-all-reachable-nodes-in-a-graph
        except:
            return 0

        coverage = 0
        for node in nodes_in_network:
            children = self.graph[node]
            coverage += len(children)**(1/(shortest_path_dict[S][node]+1))

        return coverage
    
    #other, deprecated influence inference algorithms
    # def firstChildrenCoverage(self, S):
    #     coverage = 0

    #     children = self.graph[S]

    #     for child in children:
    #         child_children = self.graph[child]
    #         coverage += len(child_children)**(1/2)

    #     return coverage

    # def nodeExplore(self, S, depth = 1):
    #     self.visited = [False] * (self.V)

    #     self.score = 0

    #     children = self.graph[S]

    #     for child in children:
    #         if self.visited[S] == False:
    #             self.nodeExploreUtil(child, depth = depth + 1)

    #     return self.score

    # def nodeExploreUtil(self, S, depth = 1):
    #     self.visited[S] = True

    #     children = self.graph[S]

    #     self.score += len(children)**(1/depth)

    #     for child in children:
    #         # print(child)
    #         if self.visited[S] == False:
    #             self.nodeExploreUtil(child, depth+1)
        
    #     return
    

    # def nodeExplore(self, S, depth = 0):
    #     self.visited = [False] * (self.V)

    #     self.score = 0

    #     children = self.graph[S]

    #     for child in children:
    #         if self.visited[child] == False:
    #             self.nodeExploreUtil(child, depth = depth + 1)

    #     return self.score

    # def nodeExploreUtil(self, S, depth = 0):
    #     self.visited[S] = True

    #     children = self.graph[S]

    #     self.score += len(children)**(1/depth)

    #     for child in children:
    #         # print(child)
    #         if self.visited[child] == False:
    #             self.nodeExploreUtil(child, depth+1)
        
    #     return
    
    # def edgeExplore(self, S, depth = 0):
    #     self.edges = []

    #     self.score = 0

    #     children = self.graph[S]

    #     for child in children:
    #         self.edges.append((S, child))
    #         self.edgeExploreUtil(child, depth = depth+1)

    #     return self.score

    # def edgeExploreUtil(self, S, depth = 0):
    #     children = self.graph[S]

    #     # print(S, 1/(depth+1))
    #     self.score += 1/(depth+1)

    #     for child in children:
    #         #if the edge hasn't already been mapped and you're not going back a layer (child is in source node)
    #         if (S, child) not in self.edges and child not in [edge[0] for edge in self.edges]:
    #             # print((S, child))
    #             self.edges.append((S, child))

    #             self.edgeExploreUtil(child, depth = depth+1)
        
    #     return

def create_miRNA_graph(disease, network = None, path_to_network = None):
    """
    Creates miRNA_graph, disease specific miRNA graph and mapping from graph keys to miRNA names
    """

    if path_to_network is not None:
        network = pd.read_csv(path_to_network, header=None, low_memory=False)

        #preprocesses network
        del network[0]
        network.columns = network.iloc[0]
        network = network.iloc[1:]
        network.index = network.columns
        network = network.apply(pd.to_numeric, errors='coerce') #https://stackoverflow.com/questions/45478070/pd-read-csv-gives-me-str-but-need-float
    elif network is not None:
        pass
    else:
        raise IOError("No network in valid format was passed")
    
    disease_associated = get_HMDD_disease_associated_db(disease)

    miRNA_graph = Graph(len(network.index))
    disspec_miRNA_graph = Graph(len(network.index))

    key_to_miRNA_name_dict = {} #maps the key in the miRNA_graph to the miRNA name

    miRNA_names = list(network.columns)

    network = network.to_dict('records')
    
    #https://towardsdatascience.com/heres-the-most-efficient-way-to-iterate-through-your-pandas-dataframe-4dad88ac92ee
    for row_count, row in tqdm(enumerate(network)):
        key_to_miRNA_name_dict[row_count] = str(miRNA_names[row_count]).lower()

        for col_count, col in enumerate(row):
            if row[col] > 0:
                miRNA_graph.addEdge(row_count, col_count)

                regulator = re.sub('(-5p|-3p|.3p|.5p)$', '', str(miRNA_names[row_count]).lower())
                target = re.sub('(-5p|-3p|.3p|.5p)$', '', str(miRNA_names[col_count]).lower())

                if regulator in (mir.lower() for mir in disease_associated['mir']) and target in (mir.lower() for mir in disease_associated['mir']):
                    disspec_miRNA_graph.addEdge(row_count, col_count)

    return miRNA_graph, disspec_miRNA_graph, key_to_miRNA_name_dict

