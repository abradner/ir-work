import matplotlib.pyplot as plt
import numpy as np

from drawille import Canvas
from gather_raw_codes import RawCodeGatherer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import socket

def listen_for_data():
    """
    Open a TCP socket and listen for data from another process.
    Each message is a pair of integers, the first representing the pulse length and the second representing the space length.
    """

    # Create a TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('localhost', 10000)
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    # Wait for a connection
    connection, client_address = sock.accept()

    # Receive data
    while True:
        data = connection.recv(1024)
        if not data:
            break

        # Convert the data to a list of integers
        data = data.decode('utf-8').split(',')
        data = [int(x) for x in data]

        yield data

    # Close the connection
    connection.close()

def raw_data_to_graph():
    """
    Build a NP array of pulse/space pairs. 
    
    every 100ms we send all the data to the graph

    """

def discover_encoding_structure_candidates(data, max_clusters=10):
    """
    Attempt to discover the encoding structure of the underlying signal.

    Args:
        data (dict): A nested dictionary containing the gathered data for each button and length.
        max_clusters (int): The maximum number of clusters to consider.

    Returns:
        dict: A dictionary where keys are the number of clusters and values are the corresponding cluster centers.
    """
    # Flatten the data into a single list of (pulse, space) pairs
    all_data = []
    for button_data in data.values():
        for length_data in button_data.values():
            all_data.extend(length_data)

    # Convert the data to a numpy array for use with sklearn
    all_data = np.array(all_data)

    # Dictionary to hold the cluster centers for each number of clusters
    clusterings = {}

    # Try different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_data)
        clusterings[n_clusters] = kmeans.cluster_centers_.tolist()

    # Rank the clusterings by silhouette score
    # The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters
    # The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters
    ranked_clusterings = sorted(clusterings.items(), key=lambda item: silhouette_score(all_data, KMeans(n_clusters=item[0]).fit_predict(all_data)), reverse=True)

    return dict(ranked_clusterings)

def normalize(value, min_value, max_value, new_min, new_max):
    # Normalize value to [0, 1]
    value = (value - min_value) / (max_value - min_value)

    # Scale to new range
    value = value * (new_max - new_min) + new_min

    return int(value)

def plot_clusters(data, clusters):
    """
    Plot the data and the clusters.

    Args:
        data (dict): The gathered data.
        clusters (dict): The clusters.
    """
    # Flatten the data into a single list of (pulse, space) pairs
    all_data = []
    for button_data in data.values():
        for length_data in button_data.values():
            all_data.extend(length_data)

    # Convert the data to a numpy array for use with matplotlib
    all_data = np.array(all_data)

    # Create a scatter plot of the data
    plt.scatter(all_data[:, 0], all_data[:, 1])

    # Plot the cluster centers
    for n_clusters, cluster_centers in clusters.items():
        cluster_centers = np.array(cluster_centers)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x')

    # Determine the min and max values for x and y
    min_x, max_x = np.min(all_data[:, 0]), np.max(all_data[:, 0])
    min_y, max_y = np.min(all_data[:, 1]), np.max(all_data[:, 1])

    # Convert the plot to a text plot
    canvas = Canvas()


    # Plot the x-axis
    for x in range(80):
        canvas.set(x, 40)

    # Plot the y-axis
    for y in range(80):
        canvas.set(40, y)


    for x, y in zip(all_data[:, 0], all_data[:, 1]):
        # Normalize the x and y values to fit within the desired range
        x = normalize(x, min_x, max_x, 0, 80)
        y = normalize(y, min_y, max_y, 0, 80)
        canvas.set(x, y)


    # Print the text plot
    print(canvas.frame())
