import subprocess
import numpy as np
from sklearn.cluster import KMeans
import json
import os
import sys

def parse_signal(signal, header_pulse, header_space, zero_pulse, zero_space, one_pulse, one_space):
    """Convert raw signal into hex code."""
    code = ""
    # Check if the first pulse and space match the header values
    if signal[0] == header_pulse and signal[1] == header_space:
        for i in range(2, len(signal), 2):
            if signal[i] == one_pulse and signal[i + 1] == one_space:
                code += "1"
            elif signal[i] == zero_pulse and signal[i + 1] == zero_space:
                code += "0"
    return code

def tune_ir_structure(raw_signal):
    """Tune the IR signal structure based on the raw signal."""
    # Convert raw signal to numpy array
    raw_signal = np.array(raw_signal).reshape(-1, 1)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(raw_signal)

    # Calculate the average of each cluster
    averages = [np.mean(raw_signal[kmeans.labels_ == i]) for i in range(2)]

    # Sort averages
    averages.sort()

    # Return the averages as the new zero and one pulse/space durations
    return averages[0], averages[1]

def extract_pulses_and_spaces(raw_signal):
    """Extract the pulses and spaces from the raw signal."""
    pulses = [value for signal_type, value in raw_signal if signal_type == 'pulse']
    spaces = [value for signal_type, value in raw_signal if signal_type == 'space']
    return pulses, spaces


def decode_signal(raw_signal, zero_pulse, one_pulse, zero_space, one_space, tolerance):
    """Decode the signal into binary and hexadecimal codes."""
    code = ""
    codes = []
    for i in range(0, len(raw_signal), 2):
        signal_type, value = raw_signal[i]
        next_signal_type, next_value = raw_signal[i + 1] if i + 1 < len(raw_signal) else (None, None)
        if signal_type == 'timeout':
            # Convert the binary code to hexadecimal and add it to the list of codes
            codes.append(hex(int(code, 2)))
            code = ""  # Start a new code
        elif signal_type == 'pulse' and next_signal_type == 'space':
            if abs(value - one_pulse) <= tolerance and abs(next_value - one_space) <= tolerance:
                code += "1"
            elif abs(value - zero_pulse) <= tolerance and abs(next_value - zero_space) <= tolerance:
                code += "0"
            else:
                tolerance += 50  # Increase the tolerance by 50 each time an unrecognized pair is encountered
                print(f"Warning: Unrecognized signal pair: {value}, {next_value}. Increasing tolerance to {tolerance}.")

    # Add the last code to the list of codes
    if code:
        codes.append(hex(int(code, 2)))

    return codes

def process_signal(raw_signal, tuning_data):
    """Process a raw signal and update the tuning data."""
    pulses, spaces = extract_pulses_and_spaces(raw_signal)
    zero_pulse, one_pulse = calculate_pulse_and_space(pulses)
    zero_space, one_space = calculate_pulse_and_space(spaces)
    codes = decode_signal(raw_signal, zero_pulse, one_pulse, zero_space, one_space, tuning_data['tolerance'])

    print(f"Decoded signals: {codes}")

    return tuning_data

