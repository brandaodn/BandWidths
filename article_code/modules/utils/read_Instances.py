import math
import sys
from collections import defaultdict
from scipy.sparse import lil_matrix


def load_instance(filename):
    """Load either Matrix Market coordinate data or the legacy 2-column edge-list format.

    Supported headers:
      - Matrix Market-like: nrows ncols nnz
      - Legacy graph format: nnodes nedges

    Remaining rows may contain two or more columns; only the first two are used
    as vertex indices. Self-loops are ignored for bandwidth purposes.
    """
    edges = []
    neighbours = defaultdict(list)
    seen_edges = set()

    with open(filename, "r") as f:
        data_lines = []
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("%"):
                continue
            data_lines.append(parts)

    if not data_lines:
        raise ValueError(f"Empty instance: {filename}")

    header = data_lines[0]
    if len(header) >= 3:
        nrows, ncols, declared_nnz = map(int, header[:3])
        if nrows != ncols:
            raise ValueError(
                f"Bandwidth benchmark expects a square matrix, got {nrows}x{ncols}: {filename}"
            )
        nnodes = nrows
    elif len(header) == 2:
        nnodes, declared_nnz = map(int, header)
    else:
        raise ValueError(f"Invalid instance header in {filename}: {header}")

    for parts in data_lines[1:]:
        if len(parts) < 2:
            continue
        e1, e2 = int(float(parts[0])), int(float(parts[1]))
        if e1 < 1 or e2 < 1 or e1 > nnodes or e2 > nnodes:
            raise ValueError(
                f"Vertex index out of range in {filename}: ({e1}, {e2}) for n={nnodes}"
            )
        if e1 == e2:
            continue

        edge = (min(e1, e2), max(e1, e2))
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        edges.append(edge)
        neighbours[e1].append(e2)
        neighbours[e2].append(e1)

    # Preserve isolated vertices explicitly.
    for node in range(1, nnodes + 1):
        neighbours[node]

    nedges = len(edges)
    nodes = list(range(1, nnodes + 1))
    lista_adj = [neighbours[n] for n in nodes]

    matrix = lil_matrix((nnodes, nnodes), dtype=int)
    for u, v in edges:
        matrix[u - 1, v - 1] = 1
        matrix[v - 1, u - 1] = 1

    return nnodes, nedges, edges, neighbours, lista_adj, matrix


def load_instance_x(filename):
    #ler as instancias thermais
    edges = []
    neighbours = defaultdict(list)
    neigh = defaultdict(list)
    flag = True
    f = open(filename, 'r')
    for line in f:

        if flag == True:
            nnodes, value, nedges = [int(x) for x in line.split()]
            flag = False
        else:
            e1, e2 = [x for x in line.split()]
            e1 = int(e1)
            e2 = int(e2)
            neigh[e1].append(e2)
            if e1 != e2:
                edges.append((min(e1, e2), max(e1, e2)))
                neighbours[e1].append(e2)
                neighbours[e2].append(e1)
            else:
                nedges = nedges - 1
    f.close()

    f = []

    for v in neighbours:
        f.append(v)

    nodes = []

    for v in neighbours:
        nodes.append(v)

    lista_adj = []

    for n in nodes:
        lista_adj.append(neighbours[n])

    matrix = []

    for key in neighbours:
        line = [0]*nnodes
        for element in neighbours[key]:
            line[element-1] = 1
        matrix.append(line)

    return nnodes, nedges, edges, neighbours, lista_adj, matrix


def print_instance(qtd_nodes, qtd_edges, edges, neighbours):
    print(str(qtd_nodes)+" "+str(qtd_edges))

    for e in edges:
        print(str(e[0])+" "+str(e[1]))

    for i in neighbours:
        print(neighbours[i])

    print(neighbours)
