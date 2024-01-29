import queue
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import socket

from sklearn.cluster import KMeans
from datetime import datetime, timedelta
from time import sleep

# This function handles the socket communication
def handle_socket():
  # Initialize a buffer to store the incoming data
  buffer = ""

  # Create a TCP/IP socket
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  # Bind the socket to a specific address and port
  server_address = ('0.0.0.0', 12345)
  sock.bind(server_address)

  # Listen for incoming connections
  sock.listen(1)
  print('listening to {} port {}'.format(*server_address))

  while True:
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()

    try:
      print('connection from', client_address)

      # Receive the data in small chunks and add it to the buffer
      while True:
        raw_data = connection.recv(1024)
        if not raw_data:
          break
        buffer += raw_data.decode('utf-8')

        # Process as many complete messages as are available in the buffer
        while '\n' in buffer:
          pair_str, sep, buffer = buffer.partition('\n')
          x_str, sep, y_str = pair_str.partition(',')
          x, y = int(x_str), int(y_str)

          # Add the data to the queue
          data_queue.put((x, y))

    finally:
      # Clean up the connection
      print("Closing connection")
      connection.close()
      print("Closing socket")
      sock.close()
      print("Network thread finished")

# This function is called periodically to update the plot
def update_graph():
  buffer = ""

  while True:
    x, y = data_queue.get()  # This will block until an item is available
    data_points.append((x, y))

    if x < 700:
      # if y > 2000:
      #   pass
      #   # group_counts[2] += 1
      # else:
      group_counts[0] += 1
      buffer += "0"
    else:
      group_counts[1] += 1
      buffer += "1"
    # print(group_counts)

    if len(buffer) == 12:
      output_queue.put(buffer)
      buffer = ""

    ax.scatter(x, y)

def calculate_clusters():
  last_cluster_time = datetime.now() - timedelta(seconds=1)
  last_data_point_count = 0

  while True:
    time_diff = datetime.now() - last_cluster_time
    refresh_by_time = time_diff >= timedelta(seconds=1)

    len_data = len(data_points)
    refresh_by_count = len_data > 20 and len_data > last_data_point_count
    last_data_point_count = len_data

     # Perform clustering at most once every second (once we have enough data points)
    if refresh_by_time and refresh_by_count:
      last_cluster_time = datetime.now()

      # Use the elbow method to determine the optimal number of clusters
      distortions = []
      for i in range(2, 6):
        kmeans = KMeans(n_clusters=i, random_state=0, n_init=10).fit(data_points)
        distortions.append(kmeans.inertia_)

      # Find the elbow point
      elbow_point = distortions.index(min(distortions)) + 1

      # Perform KMeans clustering with the optimal number of clusters
      kmeans = KMeans(n_clusters=elbow_point, random_state=0, n_init=10).fit(data_points)

      # round the cluster centers to the nearest integer
      cluster_centers = [[round(x), round(y)] for x, y in kmeans.cluster_centers_]

      # for each cluster centre, count the number of data points that are closest to it
      labels = kmeans.labels_
      cluster_centers_with_counts = []
      for i, center in enumerate(cluster_centers):
          count = sum(labels == i)
          cluster_centers_with_counts.append((center, count))

      # Print and plot the cluster centers
      print("Cluster centers:\n", cluster_centers_with_counts)

      for center in cluster_centers:
        ax.scatter(center[0], center[1], c='red', marker='x')
    else:
      sleep(0.5)


def show_output():
  last_output = ""
  while True:
    res = output_queue.get()
    if res != last_output:
      last_output = res
      res_hex = hex(int(res, 2))
      print(f"Received: {res} ({res_hex})")

# Initialize a queue to store the incoming data
data_queue = queue.Queue()
output_queue = queue.Queue()
data_points = []

# Create the figure and axis objects
fig, ax = plt.subplots()

# Set the x and y limits of the plot
plt.xlim(0, 2000)
plt.ylim(0, 10000)

# Turn on interactive mode
plt.ion()

# Start the socket communication in a separate thread
socket_thread = threading.Thread(target=handle_socket)
socket_thread.start()

# Start the graph update in a separate thread
threading.Thread(target=update_graph, daemon=True).start()

# Start the graph update in a separate thread
threading.Thread(target=show_output, daemon=True).start()


# Start the graph update in a separate thread
threading.Thread(target=calculate_clusters, daemon=True).start()

# Start the main loop
plt.show(block=False)

group_counts = [0,0,0]
old_group_counts = [0,0,0]

try:
  while True:
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # if group_counts != old_group_counts:
    #   print(group_counts)
    #   old_group_counts = group_counts.copy()

    plt.pause(0.1)
except KeyboardInterrupt:
  print("Exiting")
  socket_thread.join()
  exit()
