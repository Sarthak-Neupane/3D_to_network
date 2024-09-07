import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize a Network/Graph
G = nx.Graph()

# Example 3D coordinates
coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

# Add nodes with positions
for i, coord in enumerate(coordinates):
    G.add_node(i, pos=coord)

# Add edges based on Euclidean distance (connecting to nearest neighbors)
for i, coord1 in enumerate(coordinates):
    for j, coord2 in enumerate(coordinates):
        if i != j:
            dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
            if dist < 1.5:  # threshold for "nearest" neighbors
                G.add_edge(i, j, weight=dist)

# Plot the initial network
pos = nx.spring_layout(G, dim=3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def draw_network(axis, G, pos, node_colors):
    axis.clear()
    axis.set_axis_off()
    
    # Draw nodes
    for node, (x, y, z) in pos.items():
        axis.scatter(x, y, z, c=node_colors[node], s=100)

    # Draw edges
    for (i, j) in G.edges():
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        z = [pos[i][2], pos[j][2]]
        axis.plot(x, y, z, c='black')

    axis.set_title("Diffusion Process in 3D Network")

# Initialize Messages
firstInfected = random.choice(list(G.nodes))

# SI Model
model = ep.SIModel(G)

# Model configuration
config = mc.Configuration()
config.add_model_parameter('beta', 0.2)  # Infection probability
config.add_model_initial_configuration("Infected", [firstInfected])

model.set_initial_status(config)

# Simulation
iterations = model.iteration_bunch(150)  # Adjust the number of iterations as needed

# Animation of the diffusion process
nodeColors = ['blue'] * len(G.nodes)  # Initial color: blue for susceptible

def update(num, iterations, nodeColors):
    iteration = iterations[num]
    for node in iteration['status']:
        nodeColors[node] = 'red'  # Red for infected

    draw_network(ax, G, pos, nodeColors)

ani = animation.FuncAnimation(fig, update, frames=len(iterations), fargs=(iterations, nodeColors), repeat=False)

plt.show()
