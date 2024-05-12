# -*- ecoding: utf-8 -*-
# @ModuleName: test
# @Function: 
# @Author: wenYan(pepper)
# @Time: 2024/4/11 02:05
# Required Libraries
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Visualization function for embedding process
def visualize_embedding_process():
    # Define the vectors
    vectors = {
        'data': np.array([1, 0, 0]),
        'spe': np.array([0, 1, 0]),
        'w': np.array([0, 0, 1]),
        'd': np.array([1, 1, 0]) / np.sqrt(2),
        'h': np.array([1, 0, 1]) / np.sqrt(2),
        'tme': np.array([0, 1, 1]) / np.sqrt(2),
    }

    # Define vector colors
    colors = {
        'data': 'blue',
        'spe': 'green',
        'w': 'red',
        'd': 'purple',
        'h': 'orange',
        'tme': 'brown',
    }

    # Plotting the vectors
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.quiver(0, 0, 0, vectors['data'][0], vectors['data'][1], vectors['data'][2], color=colors['data'], scale=1, scale_units='xy', angles='xy')
    ax.quiver(vectors['data'][0], vectors['data'][1], vectors['data'][2], vectors['spe'][0], vectors['spe'][1], vectors['spe'][2], color=colors['spe'], scale=1, scale_units='xy', angles='xy')
    ax.quiver(vectors['data'][0] + vectors['spe'][0], vectors['data'][1] + vectors['spe'][1], vectors['data'][2] + vectors['spe'][2], vectors['w'][0], vectors['w'][1], vectors['w'][2], color=colors['w'], scale=1, scale_units='xy', angles='xy')
    ax.quiver(vectors['data'][0] + vectors['spe'][0] + vectors['w'][0], vectors['data'][1] + vectors['spe'][1] + vectors['w'][1], vectors['data'][2] + vectors['spe'][2] + vectors['w'][2], vectors['d'][0], vectors['d'][1], vectors['d'][2], color=colors['d'], scale=1, scale_units='xy', angles='xy')
    ax.quiver(vectors['data'][0] + vectors['spe'][0] + vectors['w'][0] + vectors['d'][0], vectors['data'][1] + vectors['spe'][1] + vectors['w'][1] + vectors['d'][1], vectors['data'][2] + vectors['spe'][2] + vectors['w'][2] + vectors['d'][2], vectors['h'][0], vectors['h'][1], vectors['h'][2], color=colors['h'], scale=1, scale_units='xy', angles='xy')
    emb_vector = sum(vectors.values())
    ax.quiver(0, 0, 0, emb_vector[0], emb_vector[1], emb_vector[2], color='black', scale=1, scale_units='xy', angles='xy')

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of Data Embedding Process')

    # Annotate vectors
    for label, vector in vectors.items():
        ax.text(vector[0], vector[1], vector[2], f'{label}', color=colors[label], size=15)

    # Annotate resulting vector
    ax.text(emb_vector[0], emb_vector[1], emb_vector[2], 'emb', color='black', size=15)

    # Adding a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=color, markersize=15) for key, color in colors.items()]
    ax.legend(handles=handles, loc='upper right')

    # Show the plot
    plt.grid(True)
    plt.show()

# Call the function to visualize the process
visualize_embedding_process()
