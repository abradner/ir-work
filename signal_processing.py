import threading
import time
import queue

import matplotlib.pyplot as plt
import numpy as np

from drawille import Canvas
from gather_raw_codes import RawCodeGatherer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class TimeoutException(Exception):
    pass

def discover_signal_encoding_and_structure():
    """
    an algorithm to attempt to discover the structure of the underlying signal. We're putting asside our assumptions that:
    - zeros and ones have distinct space lengths (spaces may be common length)
    - zeros and ones are encoded in the pulses not the spaces (hypothesis: pulses may be consistent with different length spaces)
    - the entire message is delivered in one word
    - headers are only sent once
    
    Initial idea (needs iteration) 
    1) Gather as much data as we think we need from button presses of a given length (eg short, medium, long), 
    2) potentially repeat using a different button or two in order to identify header, body, footer, message length, repeat codes, etc
    3) use this data to attepmt to discover the encoding structure, identifying gap lengths, pulse lengths, etc. 
      We should create some clusters where the clusters are the different lengths of pulses and spaces.
    4) we use those clusters to attempt to label the data (eg 0 pulse/space, 1 pulse/space, header pulse/space, footer pulse/space, etc)
      We need to be mindful that not all of these features may exist (eg spaces OR pulses may be consistent)
      We need to figure out which is a zero and which is a one from the data as well
    5) Having deterimined the encoding structure, we can then attempt to decode the data.
      We can now attempt to identify the header (if any), body, message body length, and footer (if any)
    6) We should report back the candidates for encoding structure and message structure to the user in order of likelihood of correctness
    7) allow the user to select one and then persist that data for future use
    """
    # Decide how many button lengths we want to gather data for
    lengths = [2000]#[500, 1500, 5000]  # short, mid, long

    # Decide how many buttons we want to gather data for
    num_buttons = 1  # adjust this value as needed
    # Gather data from button presses
    raw_data = gather_data_from_button_presses(num_buttons, lengths)

    # Discover encoding structure candidates
    cluster_candidates = discover_encoding_structure_candidates(raw_data)

    # Plot the data and the clusters
    plot_clusters(raw_data, cluster_candidates)

    # possible_encodings = []
    # for cluster in cluster_candidates:
    #     encoding = {}
    #     encoding['cluster'] = cluster
    #     encoding['labeled_data'] = label_data(raw_data, cluster)
    #     encoding['decoded_data'] = decode_data(raw_data, encoding['labeled_data'])
    #     possible_encodings.append(encoding)

    # report_candidates(possible_encodings)
    # encoding_structure, message_structure = select_structure(possible_encodings)
    # persist_structure(encoding_structure, message_structure)


# def gather_data_from_button_presses(num_buttons, lengths):
#     """
#     Gather data from button presses of different lengths (lengths measured in milliseconds).

#     Args:
#         buttons (list): The buttons to gather data from.
#         lengths (list): A list of lengths to gather data for.

#     Returns:
#         dict: A dictionary containing the gathered data for each length.
#     """

#     data = {}
#     rcg = RawCodeGatherer()


#     # Prompt the user asking them to push button(s) for the length(s) we choose
#     for button in range(0, num_buttons):
#         data[button] = {}
#         for length in lengths:
#             try:
#                 print(f"Please push button {button} for {length} milliseconds")
#                 gathered_data = gather_raw_button_data(rcg, length)
#             except TimeoutError: # If the RCG times out, we should retry the step until ctrl+c is pressed
#                 print("WARN: Could not read pair from mode2 output.")
#                 # retry the step once more
#                 gathered_data = gather_raw_button_data(rcg, length)
#             data[button][length] = gathered_data

#     return data

# def timeout_handler(length):
#     time.sleep(length / 1000.0)  # Convert length from milliseconds to seconds
#     raise TimeoutException()

# def gather_raw_button_data(rcg, length):
#     gathered_data = []
#     timer = threading.Thread(target=timeout_handler, args=([length]))

#     try:
#         timer.start()
#         while True:
#             gathered_data.append(rcg.gather_pair())
#     except TimeoutException:
#         pass

#     return gathered_data

class TimeoutException(Exception):
    pass

def gather_raw_button_data(rcg, q):
    while True:
        try:
            pair = rcg.gather_pair()
            q.put(pair)
        except Exception as e:
            q.put(e)
            break

def gather_data_with_timeout(rcg, length):
    q = queue.Queue()
    gather_thread = threading.Thread(target=gather_raw_button_data, args=(rcg, q))
    gather_thread.start()

    end_time = time.time() + length / 1000.0  # Convert length from milliseconds to seconds
    gathered_data = []

    while time.time() < end_time:# or not q.empty():
        try:
            pair = q.get(timeout=end_time - time.time())
            if isinstance(pair, Exception):
                raise pair
            gathered_data.append(pair)
        except queue.Empty:
            break

    return gathered_data

def gather_data_from_button_presses(num_buttons, lengths):
    data = {}
    rcg = RawCodeGatherer()

    for button in range(0, num_buttons):
        data[button] = {}
        for length in lengths:
            try:
                print(f"Please push button {button} for {length} milliseconds")
                gathered_data = gather_data_with_timeout(rcg, length)
            except TimeoutError:  # If the RCG times out, we should retry the step until ctrl+c is pressed
                print("WARN: Could not read pair from mode2 output.")
                # retry the step once more
                gathered_data = gather_data_with_timeout(rcg, length)
            data[button][length] = gathered_data

    return data

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

def label_data(data, clusters):
    """
    Label the data based on the discovered encoding structure.

    Args:
        data (dict): A dictionary containing the gathered data for each length.
        clusters (list): A list of clusters representing different lengths of pulses and spaces.

    Returns:
        dict: A dictionary containing the labeled data.
    """
    labeled_data = {}
    for length, pairs in data.items():
        # TODO: Implement the logic to assign each (pulse, space) pair to a cluster
        # The labeled_data should be a dictionary, where the keys are the same as in data and the values are lists of cluster labels
        labeled_data[length] = []
    return labeled_data

def decode_data(data, labeled_data):
    """
    Decode the data based on the labeled data.

    Args:
        data (dict): A dictionary containing the gathered data for each length.
        labeled_data (dict): A dictionary containing the labeled data.

    Returns:
        str: The decoded data.
    """
    decoded_data = {}
    for length, labels in labeled_data.items():
        # TODO: Implement the logic to decode the labels into a binary string
        # The decoded_data should be a dictionary, where the keys are the same as in data and the values are binary strings
        decoded_data[length] = ""
    return decoded_data

def report_candidates(encoding_structure, message_structure):
    """
    Report the candidates for encoding structure and message structure to the user.

    Args:
        encoding_structure (list): A list of encoding structure candidates.
        message_structure (list): A list of message structure candidates.
    """
    pass

def select_structure(encoding_structure, message_structure):
    """
    Allow the user to select the encoding structure and message structure.

    Args:
        encoding_structure (list): A list of encoding structure candidates.
        message_structure (list): A list of message structure candidates.

    Returns:
        tuple: A tuple containing the selected encoding structure and message structure.
    """
    pass

def persist_structure(encoding_structure, message_structure):
    """
    Persist the selected encoding structure and message structure for future use.

    Args:
        encoding_structure (list): The selected encoding structure.
        message_structure (list): The selected message structure.
    """
    pass



# DEPRECATED
# def extract_pulses_and_spaces(raw_signal):
#     """Extract the pulses and spaces from the raw signal."""
#     # print(f"Extracting pulses and spaces from raw signal")
#     pulses = [value for signal_type, value in raw_signal if signal_type == 'pulse']
#     spaces = [value for signal_type, value in raw_signal if signal_type == 'space']
#     # print(f"=====\nExtracted pulses: {pulses}\nExtracted spaces: {spaces}\n=====")
#     return pulses, spaces

# DEPRECATED
# def decode_signal(raw_signal, tuning_data):
#     """Decode the signal into binary and hexadecimal codes."""
#     zero_pulse = tuning_data['zero_pulse']
#     one_pulse = tuning_data['one_pulse']
#     zero_space = tuning_data['zero_space']
#     one_space = tuning_data['one_space']
#     tolerance = tuning_data['tolerance']

#     print(f"Decoding signal with zero_pulse: {zero_pulse}, one_pulse: {one_pulse}, zero_space: {zero_space}, one_space: {one_space}, tolerance: {tolerance}")

#     code = ""
#     codes = []
#     for i in range(0, len(raw_signal), 2):
#         signal_type, value = raw_signal[i]
#         next_signal_type, next_value = raw_signal[i + 1] if i + 1 < len(raw_signal) else (None, None)
#         if signal_type == 'timeout':
#             # Convert the binary code to hexadecimal and add it to the list of codes
#             codes.append(hex(int(code, 2)))
#             code = ""  # Start a new code
#         elif signal_type == 'pulse' and next_signal_type == 'space':
#             if abs(value - one_pulse) <= tolerance and abs(next_value - one_space) <= tolerance:
#                 code += "1"
#             elif abs(value - zero_pulse) <= tolerance and abs(next_value - zero_space) <= tolerance:
#                 code += "0"
#             else:
#                 tolerance += 20  # Increase the tolerance by 50 each time an unrecognized pair is encountered
#                 tuning_data['tolerance'] = tolerance # Update the tuning data in a shit way
#                 print(f"Warning: Unrecognized signal pair: {value}, {next_value}. Increasing tolerance to {tolerance}.")

#     # Add the last code to the list of codes
#     if code:
#         codes.append(hex(int(code, 2)))

#     return codes

# DEPRECATED
# def process_signal(raw_signal, tuning_data):
#     """Process a raw signal and update the tuning data."""
#     codes = decode_signal(raw_signal, tuning_data)
#     print(f"Decoded signals: {codes}")
#     return codes

print("running")
discover_signal_encoding_and_structure()
