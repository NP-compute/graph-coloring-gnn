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

# Here we specify which datasets to use (can be arbitrary number of files):
problem_files = [
    'add_1bit_0_0.col',
    'add_1bit_1_0.col',
    'add_1bit_0_1.col',
    'add_1bit_1_1.col',
    'add_2bit_0_0.col',
    'add_2bit_1_0.col',
    'add_2bit_0_1.col',
    'add_2bit_1_1.col',
    'add_2bit_2_0.col',
    'add_2bit_0_2.col',
    'add_2bit_2_2.col',
    'add_2bit_1_2.col',
    'add_2bit_2_1.col',
    'add_2bit_3_0.col',
    'add_2bit_0_3.col',
    'add_2bit_3_3.col',
    'add_2bit_3_1.col',
    'add_2bit_1_3.col',
    'add_2bit_2_3.col',
    'add_2bit_3_2.col',
    # Add more files here as needed
]

#Here we define the hyperparameters (initial values, will be updated per graph)
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
    'graph_file': None,  # Will be set per file
    'number_colors': 3, # Number of colors in the problem, IMPORTANT NOTE: It is 3 unless the graph is broken like the repo does for processing
    'num_nodes': None # Will be set per file
}

# Variables for model reuse across files
net = None
optimizer = None

# Store results for all graphs
all_results = []

# Train on each file sequentially
for idx, problem_file in enumerate(problem_files):
    print(f'\n{"="*80}')
    print(f'Training on file {idx+1}/{len(problem_files)}: {problem_file}')
    print(f'{"="*80}\n')

    # Build the graph for this problem
    G = build_color_graph(problem_file, file_path='./data/input/COLOR/training_graphs/')

    # Update hyperparameters for this specific graph
    hypers['graph_file'] = problem_file
    hypers['num_nodes'] = len(G.nodes)

    # Initialize model on first iteration, reuse on subsequent iterations
    if net is None:
        print("Initializing model for the first time...")
        net, initial_embedding, optimizer = initialize_model(hypers)
    else:
        print("Reusing existing model with new embedding for current graph...")
        # Reinitialize embedding for new graph size (but keep the same network and optimizer)
        initial_embedding = torch.nn.Parameter(torch.randn((hypers['num_nodes'], hypers['initial_dim']),
                                                           device=hypers['device'],
                                                           dtype=hypers['dtype']))

    # Train on this graph
    probs, best_coloring, best_loss, best_cost = training(G, net, initial_embedding, optimizer, hypers, verbose=True)

    print(f'\nFor {problem_file}, the normalized error is: {best_cost/len(G.edges)}')

    # First we obtain the node_color and edge_color lists
    node_colors, edge_colors = color_lists(G, best_coloring.cpu(), color_map)

    # Here we save the graph with this coloring
    graph_name = problem_file.replace('.col', '')
    save_graph(G, node_colors, edge_colors, f'{graph_name}-gcn', path='graphs/')

    # Then we plot the graph
    # plot_graph(G, node_colors=node_colors, edge_colors=edge_colors, seed=SEED_VALUE, node_size=100, figsize=8, name=f'{graph_name} Colored')

    # Here we obtain a new coloring along with the upper bound for the chromatic number
    optimized_coloring, upper_chromatic_number = new_coloring(G, best_coloring.cpu())
    # Here we print the upper bound for the chromatic number
    print(f'The upper bound for the chromatic number is: {upper_chromatic_number}')

    # Here we create a dataframe with the results for this graph
    results = {
        'Graph': graph_name,
        'Nodes': len(G.nodes),
        'Edges': len(G.edges),
        'Density %': nx.density(G)*100,
        'Colors': 3,
        'Chromatic Number': upper_chromatic_number.item(),
        'GCN Cost': best_cost,
        'Error %': (best_cost/len(G.edges))*100,
    }

    all_results.append(results)

    print(f"Graph:{results['Graph']} | Nodes:{results['Nodes']} | Edges:{results['Edges']} | Density %:{results['Density %']:.2f} | Colors:{results['Colors']} | Chromatic Number:{results['Chromatic Number']} | GCN Cost:{results['GCN Cost']} | Error %:{results['Error %']:.2f}")

# Print summary of all results
print(f'\n{"="*80}')
print('TRAINING SUMMARY - ALL GRAPHS')
print(f'{"="*80}\n')

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))
print(f'\n{"="*80}')

