# lendo todas as instancias de uma classe

import os
import time
import pandas as pd
import networkx as nx
import torch
from modules.utils.read_filenames import readFilesInDict
from modules.graph.Grafo import GrafoListaAdj
from centralities import get_centrality_node
from agent import Agent
from enviroment import Env
from modules.utils import read_Instances
import matplotlib.pyplot as plt
from modules.utils.handle_labels import set_bandwidth
import json

# Função para plotar uma matriz esparsa
def plot_sparse_matrix(matrix, title, file_name="saida.png"):
    plt.figure(figsize=(8, 8))
    plt.spy(matrix, markersize=1)
    plt.title(title)
    plt.savefig(file_name)   # salva em arquivo
    plt.close()              # fecha a figura para não abrir

dir_list = ["optimization", "thermal", "structural", "computational_fluid_dynamics", "electromagnetics", ]

for loop in range(1,2):

    filename = './result_output_cuthill_autovetor_maior.csv'
    if loop == 1:
        df = pd.DataFrame()
    else:
        df = pd.read_csv(filename)

    list_instance, list_band, list_time, global_iteration = [], [], [], []

    base_dir = os.path.dirname(__file__)  # diretório onde está o main.py
    json_global = os.path.join(base_dir, "data", "newdata", "global_analysis_inputs.json")
    
    for kind in dir_list:
        path = os.path.join(base_dir, "data", "newdata", kind)
        list_path = readFilesInDict(path, ".mtx")

        for instancia in list_path:
            results = {}

            instance_path = os.path.basename(instancia).replace(".mtx", "")
            path_name = os.path.join(os.path.dirname(os.path.dirname(instancia)), f"plots/{instance_path}")
            # criar o diretório se não existir
            os.makedirs(path_name, exist_ok=True)
            torch_save_path = os.path.join(os.path.dirname(os.path.dirname(instancia)), "plots", f"{instance_path}", f'trained_model_{instance_path}.pth')
            json_save_path = os.path.join(os.path.dirname(os.path.dirname(instancia)), "plots", f"{instance_path}", f'analysis_inputs_{instance_path}.json')
            
            # if os.path.exists(torch_save_path):
            #     print(f"Modelo já treinado para a instância {instance_path}, pulando...")
            #     continue

            print( "####### instancia", instancia )
            nnodes, nedges, edges, neighbours, lista_adj, matrix = read_Instances.load_instance(instancia)
            
            #parametros
            max_iter = 30

            centralities = {
                    #Standard centrality measures
                    "Degree": {"func": nx.degree_centrality, "args": {}, "reverse": True}, # "reverse": False},
                    "Closeness": {"func": nx.closeness_centrality, "args": {}, "reverse": True}, # "reverse": False},
                    "Betweenness": {"func": nx.betweenness_centrality, "args": {}, "reverse": True}, # "reverse": False},
                    "Eigenvector": {"func": nx.eigenvector_centrality, "args": {"max_iter": 1000, 'tol': 1e-06}, "reverse": True},
                    
                    # Additional centrality measures
                    "Katz Centrality": {"func": nx.katz_centrality, "args": {"alpha": 0.005, "beta": 1.0, "max_iter": 1000, "tol": 1e-06}, "reverse": True},
                    "PageRank": {"func": nx.pagerank, "args": {"alpha": 0.85}, "reverse": True},
                    "Harmonic Centrality": {"func": nx.harmonic_centrality, "args": {}, "reverse": True}, # "reverse": False},
                    # "Current-flow Betweenness": {"func": nx.current_flow_betweenness_centrality, "args": {}, "reverse": True},
                    # "Approx Current-flow Betweenness": { "func": nx.approximate_current_flow_betweenness_centrality, "args": {"epsilon": 0.1, "kmax": 5000, "normalized": True}, "reverse": True}
                    }
            
            dict_connectivity = {
                #Standard connectivity measures
                "Node Connectivity": nx.node_connectivity,
                "Edge Connectivity": nx.edge_connectivity,
                "Algebraic Connectivity": nx.algebraic_connectivity,

                # Additional connectivity measures
                # "Average Node Connectivity": nx.average_node_connectivity, # média do número de caminhos independentes entre todos os pares de nós
                "Graph Density": nx.density,  # razão entre arestas existentes e possíveis
                "Average Shortest Path Length": nx.average_shortest_path_length,  # eficiência global
                # "Global Clustering Coefficient": nx.transitivity,  # tendência de formar triângulos
            }
            
            todos_movimentos = list(range(len(centralities)))
            centralities_list = list(centralities.keys())
            #instancia os graficos networkx
            G = nx.Graph()
            G.add_nodes_from(range(1, nnodes + 1))
            G.add_edges_from(edges)

            temp_bandwidth = set_bandwidth(G)
            print("Bandwidth original: ", temp_bandwidth)
            
            # Number of connected components
            num_components = nx.number_connected_components(G)
            
            # Largest connected component
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len, default=set())
            largest_size = len(largest_component)
            
            # Subgraph induced by the largest component
            largest_subgraph = G.subgraph(largest_component)
            
            # Diameter of the largest component
            diameter = nx.diameter(largest_subgraph) if largest_size else 0

            start_time = time.time()
            
            print(f"Number of connected components: {num_components}")
            print(f"Size of the largest connected component: {largest_size}")
            print(f"Diameter of the largest connected component: {diameter}")
            
            for name, func in dict_connectivity.items():
                try:
                    result = func(G)
                    print(f"{name}: {result}")
                    results[name] = result
                except Exception as e:
                    print(f"Could not compute {func.__name__}: {e}")

            #instancia os graficos own-lib
            grafo = GrafoListaAdj()
            grafo.DefinirN(nnodes,VizinhancaDuplamenteLigada=True)
            for (u, v) in edges:
                grafo.AdicionarAresta(u, v)

            #dict of centralities values for each centrality
            centralities_maps={}
            for centrality_key, centrality in centralities.items():
                print("Calculating centrality: ", centrality_key)
                centralities_maps[centrality_key] = get_centrality_node(G,  centrality)
            
            #Agent to learning and trainning
            agent = Agent(learning_rate=0.001,
                        gamma=0.9,
                        epsilon=1,
                        eps_min=0.01,
                        eps_dec=0.995,
                        n_movements=len(centralities),
                        n_actions=len(todos_movimentos),
                        n_states=4,
                        deep=True)

            env = Env(temp_bandwidth)

            state = env.get_initial_state()
            agent_movements = agent.get_movements()
            score = 0
            t=0

            local_iteration = []
            start_time = time.time()

            for i in range(max_iter):
                t = i + 2
                #Escolha da centralidade e da lista
                action = agent.choose_action(state)
                centrality = centralities_list[action]

                centrality_dict_values = centralities_maps[centrality]
                
                info = env.step(graph=grafo, centrality_values=centrality_dict_values, cent_str=centrality, centralities=centralities)
                n_step = info["n_steps"]
                grafo_solution = info["graph"]
                solution = info["solution"]
                reward = info["reward"]
                gap = info["gap"]
                bandwidth = info["bandwidth"]
                local_iteration.extend(info["iterations"])
                
                new_state = [n_step, gap, reward, bandwidth]
                
                #Treinamento do novo estado
                agent.learn(state, action, reward, new_state, done=(i == max_iter - 1))

                #score acumulado por iteração
                score += reward

                print(f">> Episode {i+1} Gap: {gap} Reward: {reward} Score: {score} Centrality: {centrality} Bandwidth: {bandwidth}")
                state = new_state

                if temp_bandwidth > bandwidth:
                    name_matrix = f"bandwidth_{bandwidth}_{centrality}"
                    file_name = os.path.basename(instancia).replace(".mtx", f"_{name_matrix}.png")
                    
                    # salvar a melhor solução
                    temp_bandwidth = bandwidth
                    # Keep the original graph fixed across heuristic evaluations.

                    ordered_nodes = sorted(G.nodes(), key=lambda node: solution[node])
                    adj_matrix_reordered = nx.to_numpy_array(G, nodelist=ordered_nodes) 
                    plot_sparse_matrix(adj_matrix_reordered, name_matrix, file_name=os.path.join(path_name, file_name))

            end_time = time.time()
            global_time = end_time - start_time
            df_iteration = pd.DataFrame(local_iteration)
            df_iteration["Instance"] = instance_path
            df_iteration["Edges"] = nedges
            df_iteration["Nodes"] = nnodes
            df_iteration["NumOfComp"] = num_components
            df_iteration["LargestCompSize"] = largest_size
            df_iteration["Diameter"] = diameter
            df_iteration["Node Connectivity"] = results.get("Node Connectivity", None)
            df_iteration["Edge Connectivity"] = results.get("Edge Connectivity", None)
            df_iteration["Algebraic Connectivity"] = results.get("Algebraic Connectivity", None)
            # df_iteration["Average Node Connectivity"] = results.get("Average Node Connectivity", None)
            df_iteration["Graph Density"] = results.get("Graph Density", None)
            df_iteration["Average Shortest Path Length"] = results.get("Average Shortest Path Length", None)
            df_iteration['Global Time (s)'] = global_time
            df_iteration_dict = df_iteration.to_dict(orient='records')

            with open(json_save_path, 'w') as f:
                json.dump(df_iteration_dict, f, indent=4)
              
            torch.save(agent.Q.state_dict(), torch_save_path)

            global_iteration.extend(df_iteration_dict)
            # todos_movimentos = list(range(len(centralities)))

            # custo_s = centrality_heuristic(graph=grafo_adj, centrality_values=centralities_maps[centralities[0]], cent_str=centralities[0], alpha=0.3, iter_max=50, centralities=centralities)
            # print("Banda Final multicetrality: ", custo_s)
    df_global = pd.DataFrame(global_iteration)
    df_global.to_csv(filename, index=False)
    print(1)