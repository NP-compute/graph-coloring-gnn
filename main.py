from utilities import chromatic_numbers, set_seed, build_color_graph, plot_graph, initialize_model, training, color_lists, color_map, new_coloring, save_graph
import networkx as nx
import pandas as pd
import torch

# Here we set the device: GPU/CPU and the type of the tensors to use
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')

# Here we fix a seed to ensure consistent results
SEED_VALUE = 8111
set_seed(SEED_VALUE)

#Here we specify which dataset to use:
problem_file = 'add_4bit_15_13.col'

# Here we build the graph for the problem
G = build_color_graph(problem_file, file_path='./data/input/COLOR/training_graphs/')

#Here we define the hyperparameters
hypers = {
    'model': 'GCN',  # Which model to use
    'initial_dim': 64,  # Dimension of the initial embedding
    'hidden_dim': 64,      # Hidden dimension of the model
    'dropout': 0.1,     # Dropout rate
    'learning_rate': 0.005, # Learning rate
    'seed': SEED_VALUE,    # Random seed
    'device': TORCH_DEVICE, # Device to use
    'dtype': TORCH_DTYPE,   # Data type to use
    'number_epochs': int(1e4*3),   # Max number training steps
    'patience': 500,             # Number early stopping triggers before breaking loop
    'graph_file': problem_file,  # Which problem is being solved
    'number_colors': 3, # Number of colors in the problem, IMPORTANT NOTE: It is 3 unless the graph is broken like the repo does for processing
    'num_nodes': len(G.nodes) # Number of nodes in the problem
}

net, initial_embedding, optimizer = initialize_model(hypers)

probs, best_coloring, best_loss, best_cost = training(G, net, initial_embedding, optimizer, hypers, verbose=True)

print(f'In this case, the normalized error is: {best_cost/len(G.edges)}')

#First we obtain the node_color and edge_color lists
node_colors, edge_colors = color_lists(G, best_coloring.cpu(), color_map)

#Here we save the graph with this coloring
save_graph(G, node_colors, edge_colors, 'bit_trainer-gcn', path = 'graphs/')

#Then we plot the graph
plot_graph(G, node_colors= node_colors, edge_colors=edge_colors, seed=SEED_VALUE, node_size = 100, figsize=8, name='Queen 5-5 Colored')

#Here we obtain a new coloring along with the upper bound for the chromatic number
optimized_coloring, upper_chromatic_number = new_coloring(G, best_coloring.cpu())
#Here we print the upper bound for the chromatic number
print(f'The upper bound for the chromatic number is: {upper_chromatic_number}')

#Here we create a dataframe with the results for the first graph
results = {
    'Graph': 'queen5-5',
    'Nodes': len(G.nodes),
    'Edges': len(G.edges),
    'Density %':nx.density(G)*100,
    'Colors': 3,
    'Chromatic Number': upper_chromatic_number.item(),
    'GCN Cost': best_cost,
    'Error %':(best_cost/len(G.edges))*100,
}

print(f"Graph:{results['Graph']} | Nodes:{results['Nodes']} | Edges:{results['Edges']} | Density %:{results['Density %']} | Colors:{results['Colors']} | Chromatic Number:{results['Chromatic Number']} | GCN Cost %:{results['GCN Cost']} | Error %:{results['Error %']}")

