import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = nx.Graph()
coordinates = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

for i, coord in enumerate(coordinates):
    G.add_node(i, pos=coord)

for i, coord1 in enumerate(coordinates):
    for j, coord2 in enumerate(coordinates):
        if i != j:
            G.add_edge(i, j)

pos = nx.spring_layout(G, dim=3)
fig = plt.figure()
axis = fig.add_subplot(111, projection='3d')

def draw_network(axis, G, pos, nodeColors):
    axis.clear()
    axis.set_axis_off()

    for node, (x, y, z) in pos.items():
        axis.scatter(x, y, z, c=nodeColors[node], s=100)

    for (i, j) in G.edges():
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        z = [pos[i][2], pos[j][2]]
        axis.plot(x, y, z, c='black')

    axis.set_title("Diffusion Process in 3D Network")

####################################################################################################
firstInfected = random.choice(list(G.nodes))

model = ep.SIModel(G)
config = mc.Configuration()
config.add_model_parameter('beta', 0.2)  # Infection probability
config.add_model_initial_configuration("Infected", [firstInfected])

model.set_initial_status(config)

iterations = model.iteration_bunch(150)
trends = model.build_trends(iterations)

# Animation of the diffusion process
nodeColors = ['green'] * len(G.nodes)

def update(num, iterations, nodeColors):
    iteration = iterations[num]
    for node in iteration['status']:
        nodeColors[node] = 'red' # I put red for the infected

    draw_network(axis, G, pos, nodeColors)

ani = animation.FuncAnimation(fig, update, frames=len(iterations), fargs=(iterations, nodeColors), repeat=False)

plt.show()
