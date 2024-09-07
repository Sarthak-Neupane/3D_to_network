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

    axis.set_title("Diffusion of Two Messages in 3D Network")

firstInfected_A = random.choice(list(G.nodes))
firstInfected_B = random.choice(list(G.nodes)) 

modelA = ep.SIModel(G)
modelB = ep.SIModel(G)

configA = mc.Configuration()
configA.add_model_parameter('beta', 0.5)
configA.add_model_initial_configuration("Infected", [firstInfected_A])

configB = mc.Configuration()
configB.add_model_parameter('beta', 0.2)
configB.add_model_initial_configuration("Infected", [firstInfected_B])

# Set the configurations
modelA.set_initial_status(configA)
modelB.set_initial_status(configB)

# Run the simulation for both models
iterationsForA = modelA.iteration_bunch(150)
iterationsForB = modelB.iteration_bunch(150)

# Animation setup
nodeColors = ['green'] * len(G.nodes)

def update(num, iterationsForA, iterationsForB, nodeColors):
    iteration_A = iterationsForA[num]
    iteration_B = iterationsForB[num]
    
    for node in iteration_A['status']:
        if iteration_A['status'][node] == 1:
            nodeColors[node] = 'red'

    for node in iteration_B['status']:
        if iteration_B['status'][node] == 1:
            # here, if the node is already infected due to message A, I color it black
            nodeColors[node] = 'black' if nodeColors[node] == 'red' else 'blue'

    draw_network(axis, G, pos, nodeColors)

ani = animation.FuncAnimation(fig, update, frames=min(len(iterationsForA), len(iterationsForB)), fargs=(iterationsForA, iterationsForB, nodeColors), repeat=False)

plt.show()
