import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def run_SIR_simulation(n, radius, tau, sigma, k, initial_infected_fraction, max_steps):
    """
    Runs the SIR simulation on a random geometric graph.

    Parameters:
    - n: Number of nodes.
    - radius: Connection radius for the random geometric graph.
    - tau: Time steps delay before an infected node becomes infectious.
    - sigma: Number of time steps a node remains infectious.
    - k: Threshold number of infectious neighbors required to infect a susceptible node.
    - initial_infected_fraction: Fraction of nodes initially infected.
    - max_steps: Maximum number of simulation steps.

    Returns:
    - G: The random geometric graph.
    - pos: Dictionary of node positions.
    - history: A list (for each time step) of a dictionary mapping nodes to their state ('S', 'I', 'R').
    """
    # Create the random geometric graph.
    G = nx.random_geometric_graph(n, radius)
    pos = nx.get_node_attributes(G, 'pos')
    
    # Initialize all nodes as Susceptible.
    states = {node: 'S' for node in G.nodes()}
    infection_time = {node: None for node in G.nodes()}
    
    # Randomly choose the initial infected nodes.
    total_nodes = n
    num_initial_infected = max(1, int(total_nodes * initial_infected_fraction))
    initial_infected = random.sample(list(G.nodes()), num_initial_infected)
    for node in initial_infected:
        states[node] = 'I'
        infection_time[node] = 0  # infection starts at time 0
    
    history = []
    
    # Simulate time steps.
    for t in range(max_steps):
        new_states = states.copy()
        for node in G.nodes():
            if states[node] == 'S':
                # Count infectious neighbors.
                infectious_neighbors = 0
                for neighbor in G.neighbors(node):
                    if states[neighbor] == 'I' and infection_time[neighbor] is not None:
                        # A neighbor is infectious if it has been infected for at least tau steps,
                        # but not longer than tau + sigma.
                        if tau <= (t - infection_time[neighbor]) < (tau + sigma):
                            infectious_neighbors += 1
                if infectious_neighbors >= k:
                    new_states[node] = 'I'
                    infection_time[node] = t
            elif states[node] == 'I':
                if infection_time[node] is not None and (t - infection_time[node]) >= (tau + sigma):
                    new_states[node] = 'R'
        states = new_states
        history.append(states.copy())
        
        # Stop early if no node is infected.
        if all(state != 'I' for state in states.values()):
            break
    
    return G, pos, history

def plot_simulation(G, pos, history, folder="simulations", step_interval=1):
    """
    Plots the simulation for selected time steps and saves the plots as images.
    
    Each node is drawn as a point with a color:
      - Blue: Susceptible
      - Red: Infected
      - Green: Recovered

    The plots are saved in the specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    steps = len(history)
    for t, state in enumerate(history):
        if t % step_interval == 0 or t == steps - 1:
            plt.figure(figsize=(6,6))
            node_colors = []
            for node in G.nodes():
                if state[node] == 'S':
                    node_colors.append('blue')
                elif state[node] == 'I':
                    node_colors.append('red')
                elif state[node] == 'R':
                    node_colors.append('green')
            nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=50)
            plt.title(f"Time Step {t}")
            filename = os.path.join(folder, f"step_{t}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    # Fixed simulation parameters from the research poster.
    n = 200                           # number of nodes
    radius = 0.15                     # connection radius for the random geometric graph
    initial_infected_fraction = 0.05  # initial fraction of infected nodes
    max_steps = 20                    # maximum simulation steps

    # Prompt the user for tau, sigma, and k.
    tau = int(input("Enter tau (time steps before an infected node becomes infectious): "))
    sigma = int(input("Enter sigma (number of time steps a node remains infectious): "))
    k = int(input("Enter k (threshold number of infectious neighbors needed): "))
    
    # Run the simulation.
    G, pos, history = run_SIR_simulation(n, radius, tau, sigma, k, initial_infected_fraction, max_steps)
    
    # Plot and save the simulation images.
    plot_simulation(G, pos, history, folder="simulations", step_interval=1)
    
    print("Simulation complete. Images saved in the 'simulations' folder.")
