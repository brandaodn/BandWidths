from modules.graph.Grafo import Grafo, GrafoListaAdj
from modules.graph.Graph_mine import Graph
import random 
from collections import defaultdict

import numpy as np
import networkx as nx

#FUNCTION TO GET BANDWIDTH  by networkx
def set_bandwidth(G) -> int:
    """Return graph bandwidth for the current NetworkX node ordering.

    This uses the classical definition max |pos(u) - pos(v)| over all edges,
    matching Bf_graph and excluding the diagonal term.
    """
    positions = {node: idx for idx, node in enumerate(G.nodes())}
    return max(
        (abs(positions[u] - positions[v]) for u, v in G.edges()),
        default=0,
    )


def simples_init_sol(graph):
    ''''
    Gerador simples de solucao para mockar testes dos algoritmos pontualmente (substitui a solucao inicial do VNS)
    '''
    solucao = {}
    rotulos_ordenados = [i for i in range(1, graph.n+1)]
    random.shuffle(rotulos_ordenados)
    for v in graph.V():
        solucao[v] = rotulos_ordenados[v-1]
         
    return solucao


def Bf_graph(graph: GrafoListaAdj, F_labels: dict)->int:
    ''''
    Entrada: - um objeto grafo, classe GrafoListaAdj
             - dicionario de rotulos
            
    return: - inteiro representando a largura de banda

    Funcao que calcula a largura de banda de um grafo
    '''
    width = 0
    for v in graph.V():
        for u in graph.N(v):
           temp = abs(F_labels[v] - F_labels[u])
           if temp > width:
                width = temp      
    return width

def Bf_node(graph: GrafoListaAdj, F_labels: dict, node_v: int)->int:
    ''''
    Entrada: - um objeto grafo, classe GrafoListaAdj
             - dicionario de rotulos
             - vertice que se deseja calcular a largura de banda
            
    return: - inteiro representando a largura de banda do vertice

    Funcao que calcula a largura de banda de um grafo
    '''
    width = 0
    for node_u in graph.N(node_v):
        temp = abs(F_labels[node_v] - F_labels[node_u])
        if temp > width:
            width = temp       
    return width

#funcao que troca um rotulo u por outro v
def swapping(solution_f, node_v, node_u):
    
    '''
    This function swap the label vertex u with label vertex v.
    u and v are vertex of the graph
    '''
    solution_swapped = solution_f.copy()
    solution_swapped[node_u], solution_swapped[node_v] = solution_f[node_v], solution_f[node_u]
    
    return solution_swapped

def max_neighbor(graph: GrafoListaAdj, F_labels: dict, node_v: int) -> (int, int):
    ''''
    Entrada: - um objeto grafo, classe GrafoListaAdj
             - dicionario de rotulos
             - vertice que se deseja calcular o vizinho com maior rotulo
            
    return: - tupla de inteiros representando o vertice e o rotulo do vertice

    Funcao que calcula a largura de banda de um grafo
    '''
    temp = [(node_u, F_labels[node_u]) for node_u in graph.N(node_v)]
    max_tuple = max(temp, key=lambda tup: tup[1])        
    return max_tuple 

def min_neighbor(graph: GrafoListaAdj, F_labels: dict, node_v: int) -> (int, int):
    ''''
    Entrada: - um objeto grafo, classe GrafoListaAdj
             - dicionario de rotulos
             - vertice que se deseja calcular o vizinho com menor rotulo
            
    return: - tupla de inteiros representando o vertice e o rotulo do vertice

    Funcao que calcula a largura de banda de um grafo
    '''
    temp = [(node_u, F_labels[node_u]) for node_u in graph.N(node_v)]
    min_tuple = min(temp, key=lambda tup: tup[1])        
    return min_tuple

def get_mid_value(graph: GrafoListaAdj, F_labels: dict, node_v: int) -> int:
    
    (max_node, max_label) = max_neighbor(graph, F_labels, node_v)
    (min_node_, min_label) = min_neighbor(graph, F_labels, node_v)
    
    mid_value = (max_label+min_label)//2
    return mid_value
    
def get_mid_neighborhood(graph: GrafoListaAdj, F_labels: dict, node_v: int) -> list:
    mid_delta = abs(get_mid_value(graph, F_labels ,node_v) - F_labels[node_v])
    # print("mid_delta: ", mid_delta)
    #for u in graph.N(node_v):
        #print("(vert, rotulo): ",(u,F_labels[u]))
    mid_neighborhood = [ node_u for node_u in graph.N(node_v) if abs(get_mid_value(graph, F_labels ,node_v) - F_labels[node_u]) < mid_delta ]
    return mid_neighborhood

def get_n_critical_edges(graph: GrafoListaAdj, F_labels: dict, node_u: int, node_v: int, bandwidth: int )->int:
    count = 0

    # bandwidth = Bf_graph(graph, F_labels)

    #caso em que a uv é critica
    if abs(F_labels[node_u] - F_labels[node_v]) == bandwidth:
        count=count+1
    
    #arestas vizinhas de node_v tirando a aresta uv
    for u in graph.N(node_v):
        if (bandwidth == abs(F_labels[u] - F_labels[node_v])) and (u != node_u):
            count=count+1
    
    #arestas vizinhas de node_u tirando a aresta uv
    for v in graph.N(node_u):
        if bandwidth == abs(F_labels[v] - F_labels[node_u]) and (v != node_v):
            count=count+1

    return count

def get_n_critical_edges_by_node(graph: GrafoListaAdj, F_labels: dict, node_v: int, bandwidth: int )->int:
    count = 0
    #arestas vizinhas de node_v tirando a aresta
    for u in graph.N(node_v):
        if (bandwidth == abs(F_labels[u] - F_labels[node_v])):
            count=count+1
    return count

def Vc(graph: GrafoListaAdj, solution_f: dict)->int:
    max_custo = 0
    c = 0
    for i in graph.V():
        for y in graph.N(i):
            custo = abs(solution_f[i] - solution_f[y])
            if max_custo == custo:
                c += 1
            if max_custo < custo:
                c = 1
                max_custo = custo
    return c

def rho(solution_f1:dict, solution_f2:dict)->int:
    '''
    rho é uma função que calcula a distancia de Hamming entre duas soluções f e f'.

    -----------

    F_labels é um dicionario de Labels em que a key é o vertice e o value é o label

    '''
    distance = 0

    for v in list(solution_f1.keys()):
        if solution_f1[v] != solution_f2[v]:
            distance = distance + 1
    return distance

def move(graph: GrafoListaAdj, solution_f1:dict, solution_f2:dict, alpha:int)->bool:
    Band_f1 = Bf_graph(graph, solution_f1)
    Band_f2 = Bf_graph(graph, solution_f2)

    n_critical_f1 = Vc(graph=graph, solution_f=solution_f1)
    n_critical_f2 = Vc(graph=graph, solution_f=solution_f2)

    distance_rho = rho(solution_f1=solution_f1, solution_f2=solution_f2)

    if (Band_f1>Band_f2) or ((Band_f2==Band_f1) and (abs(n_critical_f1)>abs(n_critical_f2))) or ((Band_f2==Band_f1) and distance_rho>alpha):
        return True

def reconstruct_graph_by_labels(graph: GrafoListaAdj, F_labels: dict) -> GrafoListaAdj:
    """
    Rebuilds a new graph where each original vertex `v` is relabeled to `F_labels[v]`.
    All edges are mapped accordingly: an original edge (u, v) becomes (F_labels[u], F_labels[v]).

    Args:
        graph (GrafoListaAdj): Original graph with vertices 1..n.
        F_labels (dict): Mapping from original vertex to new label (typically 1..n),
                         e.g., the solution returned by `constructive.init_Solution_Centrality_lcr`.

    Returns:
        GrafoListaAdj: New graph with vertices 1..n ordered by labels and edges remapped.
    """
    # Preserve orientation setting
    new_g = GrafoListaAdj(graph.orientado)
    new_g.DefinirN(graph.n)

    # Iterate unique edges in the original graph and remap to new labels
    for u, v in graph.E():
        # `graph.E()` yields (u, v) with `u < v` for undirected graphs, avoiding duplicates
        new_u = F_labels[u]
        new_v = F_labels[v]
        new_g.AdicionarAresta(new_u, new_v)

    return new_g