#regular imports
import numpy as np
import random
import time
import networkx as nx

#import models
from modules.utils import handle_labels
from modules.graph.Grafo import Grafo, GrafoListaAdj

def get_LCR(queue:list, chosen_centrality:dict, random_centrality:str, alpha:float, centralities:dict):
    """
    Constructs a Restricted Candidate List (RCL) by sorting vertices based on centrality 
    and partitioning them into restricted and complementary sets.
    This function implements the RCL creation step commonly used in GRASP (Greedy Randomized 
    Adaptive Search Procedure) algorithms. It sorts vertices according to a chosen centrality 
    metric, selects the top alpha% as restricted candidates, and randomly shuffles both 
    the restricted and complementary sets before combining them.
    Args:
        queue (list): List of vertex identifiers to be processed.
        chosen_centrality (dict): Dictionary mapping each vertex to its centrality value.
        random_centrality (str): Key identifying which centrality metric is being used.
        alpha (float): Parameter between 0 and 1 that determines the proportion of top-ranked 
                      vertices to include in the restricted candidate list (0.0 to 1.0).
        centralities (dict): Dictionary containing centrality metric configurations, including 
                           a "reverse" boolean flag indicating sort order for the chosen metric.
    Returns:
        list: A shuffled list of vertices starting with randomized high-centrality vertices 
              (restricted set) followed by randomized remaining vertices (complementary set).
    """
    reverse_flag = centralities[random_centrality]["reverse"]

    vertices_ordenados = sorted(
        queue,
        key=lambda x: chosen_centrality[x],
        reverse=reverse_flag
    )
    qtd_elem = int(len(vertices_ordenados) * alpha)
    LCR = vertices_ordenados[:qtd_elem]
    LCR_comp = vertices_ordenados[qtd_elem:]
    random.shuffle(LCR)
    random.shuffle(LCR_comp)
    return LCR + LCR_comp

def init_Solution(graph: GrafoListaAdj)->dict:
    ''''FUNCIONANDO PROFUNDAMENTE'''
    f = {}

    for i in graph.V():
        f[i] = None

    mark = [] #vertices visitados

    queue = [] #representa a sequência de vértices encontrada pelo algoritmo de busca em profundidade

    for i in range(len(list(graph.V()))+1): #popula a lista mark de False
        mark.append(False)

    k = random.choice(list(graph.V()))

    mark[k] = True #visita o vertice escolhido
    
    queue.append(k)

    ql = 1 #O numero de vertices conectado ao vertice do nivel atual 
    f[k] = 1 #O primeiro vertice rotulado com o rotulo 1, com k escolhido aleatoriamente
    l = 0 #
    
    while l < graph.n:
        r = 0
        stack = []       
        for i1 in range(ql):
            i = queue[i1]
            for j1 in graph.N(i):
                vert = j1
                if not mark[vert]:
                    r = r + 1
                    stack.append(vert)
                    mark[vert] = True
        

        h = random.randint(0, ql-1)
        
        for j1 in range(ql):
            n = queue[h]
            l = l + 1
            f[n] = l
            h = h + 1
            if h >= ql:
                h = 0
        
        queue = stack
        ql = r
        
    return f


def init_Solution_Centrality(graph: GrafoListaAdj, edges:list, heuristic)->dict:
    
    # calculando a centralidade de cada vertice
    if heuristic != None:
        G = nx.Graph()
        G.add_edges_from(edges)
        resp_centrality = heuristic[0](G)
        nodes_centrality = {}
        for node, centrality in resp_centrality.items():
            nodes_centrality[node] = centrality
    

    f = {}

    for i in graph.V():
        f[i] = None

    mark = [] #vertices visitados

    queue = [] #representa a sequência de vértices encontrada pelo algoritmo de busca em profundidade

    for i in range(len(list(graph.V()))+1): #popula a lista mark de False
        mark.append(False)

    k = random.choice(list(graph.V()))

    mark[k] = True #visita o vertice escolhido
    
    queue.append(k)

    ql = 1 # O numero de vertices conectado ao vertice do nivel atual 
    f[k] = 1 #o primeiro vertice rotulado com o rotulo 1, com k escolhido aleatoriamente
    l = 0

    while l < graph.n:
        r = 0
        stack = []       
        for i1 in range(ql):
            i = queue[i1]
            for j1 in graph.N(i):
                vert = j1
                if not mark[vert]:
                    r = r + 1
                    stack.append(vert)
                    mark[vert] = True
        
        
        vertices_ordenados = sorted(queue, key=lambda x: nodes_centrality[x], reverse=heuristic[1])
        
        h = 0
        for j1 in range(ql):
            n = vertices_ordenados[h]
            l = l + 1
            f[n] = l
            h = h + 1

        queue = stack
        ql = r
        
    return f

def init_Solution_Centrality_otm(graph: GrafoListaAdj, nodes_centrality:dict)->dict:
    
    # calculando a centralidade de cada vertice

    f = {}

    for i in graph.V():
        f[i] = None

    mark = [] #vertices visitados

    queue = [] #representa a sequência de vértices encontrada pelo algoritmo de busca em profundidade

    for i in range(len(list(graph.V()))+1): #popula a lista mark de False
        mark.append(False)

    k = random.choice(list(graph.V()))

    mark[k] = True #visita o vertice escolhido
    
    queue.append(k)

    ql = 1 # O numero de vertices conectado ao vertice do nivel atual 
    f[k] = 1 #o primeiro vertice rotulado com o rotulo 1, com k escolhido aleatoriamente
    l = 0

    random_centrality = nodes_centrality 

    while l < graph.n:
        r = 0
        stack = []       
        for i1 in range(ql):
            i = queue[i1]
            for j1 in graph.N(i):
                vert = j1
                if not mark[vert]:
                    r = r + 1
                    stack.append(vert)
                    mark[vert] = True
        

        vertices_ordenados = sorted(queue, key=lambda x: random_centrality[x], reverse=True)
        
        
        h = 0
        for j1 in range(ql):
            n = vertices_ordenados[h]
            l = l + 1
            f[n] = l
            h = h + 1

        queue = stack
        ql = r
        
    return f

def init_Solution_Centrality_lcr(graph: GrafoListaAdj, nodes_centrality:dict, random_centrality:str, alpha:float, centralities:dict, start_node=None)->dict:
    
    # calculando a centralidade de cada vertice

    f = {}

    for i in graph.V():
        f[i] = None

    mark = [] #vertices visitados

    queue = [] #representa a sequência de vértices encontrada pelo algoritmo de busca em profundidade

    for i in range(len(list(graph.V()))+1): #popula a lista mark de False
        mark.append(False)
    
    # Choose the starting vertex. By default, retain the original
    # centrality-weighted sampling. Experiments may provide a fixed
    # pseudo-peripheral start node to separate BFS-start effects from
    # within-level centrality ordering.
    if start_node is not None:
        k = start_node
    else:
        vertices = list(nodes_centrality.keys())
        values = np.asarray([nodes_centrality[v] for v in vertices], dtype=float)
        valid = (
            len(vertices) > 0
            and np.all(np.isfinite(values))
            and np.all(values >= 0.0)
            and float(values.sum()) > 0.0
        )
        if valid:
            probs = (values / values.sum()).tolist()
            k = random.choices(vertices, probs)[0]
        else:
            # Uniform fallback for zero, negative, NaN or infinite weights.
            k = random.choice(vertices)

    mark[k] = True #visita o vertice assinado nno v_zero
    
    queue.append(k)

    ql = 1 # O numero de vertices conectado ao vertice do nivel atual 
    f[k] = 1 #o primeiro vertice rotulado com o rotulo 1, com k escolhido aleatoriamente
    l = 0

    while l < graph.n:
        r = 0
        stack = []       
        for i1 in range(ql):
            i = queue[i1]
            for j1 in graph.N(i):
                vert = j1
                if not mark[vert]:
                    r = r + 1
                    stack.append(vert)
                    mark[vert] = True
        
        array_LCR = get_LCR(queue=queue, chosen_centrality=nodes_centrality, random_centrality=random_centrality, alpha=alpha, centralities=centralities)
        
        h = 0
        for j1 in range(ql):
            n = array_LCR[h]
            l = l + 1
            f[n] = l
            h = h + 1

        queue = stack
        ql = r
        
    return f