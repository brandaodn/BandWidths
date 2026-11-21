import random
import numpy as np
import networkx as nx

def proposal(action,order,edges,rng):
    n=len(order);pos={v:i for i,v in enumerate(order)}
    if action==0:return rng.sample(range(n),2)
    span=max(abs(pos[u]-pos[v]) for u,v in edges)
    critical=[(u,v) for u,v in edges if abs(pos[u]-pos[v])==span]
    if not critical:return rng.sample(range(n),2)
    u,v=rng.choice(critical)
    i,j=sorted((pos[u],pos[v]))
    if action==1:
        # move one endpoint to near the midpoint, swapping labels
        k=(i+j)//2
        return (rng.choice([i,j]),k)
    if action==2:
        # move an endpoint toward the other end within +/- 2 labels
        anchor=rng.choice([i,j]); step=max(1,(j-i)//3)
        target=anchor+step if anchor==i else anchor-step
        return (anchor,max(0,min(n-1,target+rng.randint(-2,2))))
    # action 3: adjacent swap near a critical edge endpoint
    anchor=rng.choice([i,j]); delta=rng.choice([-2,-1,1,2]);return anchor,max(0,min(n-1,anchor+delta))

def instance(name,kind,n,seed):
    rng=np.random.default_rng(seed)
    if kind=='ER':g=nx.fast_gnp_random_graph(n,3.5/n,seed=seed)
    elif kind=='BA':g=nx.barabasi_albert_graph(n,2,seed=seed)
    elif kind=='WS':g=nx.watts_strogatz_graph(n,4,.12,seed=seed)
    elif kind=='grid':
        x=max(3,int(np.sqrt(n)));g=nx.grid_2d_graph(x,int(np.ceil(n/x)));g=nx.convert_node_labels_to_integers(g)
    else:
        g=nx.random_geometric_graph(n,.16,seed=seed)
    return g

