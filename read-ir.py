import subprocess
import numpy as np
from sklearn.cluster import KMeans

def parse_signal(signal, header_pulse, header_space, zero_pulse, zero_space, one_pulse, one_space):
    """Convert raw signal into hex code."""
    code = ""
    for i in range(0, len(signal), 2):
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
    kmeans = KMeans(n_clusters=2, random_state=0).fit(raw_signal)

    # Calculate the average of each cluster
    averages = [np.mean(raw_signal[kmeans.labels_ == i]) for i in range(2)]

    # Sort averages
    averages.sort()

    # Return the averages as the new zero and one pulse/space durations
    return averages[0], averages[1]

def read_mode2_output(header_pulse, header_space, zero_pulse, zero_space, one_pulse, one_space, bit_count):
    """Read output from mode2 command and process it."""
    process = subprocess.Popen(['mode2'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_signal = []
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) == 2:
                _, value = parts
                raw_signal.append(int(value))
                if len(raw_signal) >= bit_count * 2:
                    code = parse_signal(raw_signal, header_pulse, header_space, zero_pulse, zero_space, one_pulse, one_space)
                    if code == "":
                        print("No valid signal detected. Tuning IR signal structure...")
                        zero_pulse, one_pulse = tune_ir_structure(raw_signal)
                        zero_space, one_space = tune_ir_structure(raw_signal[1::2])  # Only consider space durations
                        print(f"Tuned zero_pulse: {zero_pulse}, one_pulse: {one_pulse}, zero_space: {zero_space}, one_space: {one_space}")
                        code = parse_signal(raw_signal, header_pulse, header_space, zero_pulse, zero_space, one_pulse, one_space)
                    print(f"Decoded IR Code: {hex(int(code, 2))}")
                    raw_signal = []
    except KeyboardInterrupt:
        process.kill()

# Define the IR signal structure
header_pulse = 9000
header_space = 4500
zero_pulse = 563
zero_space = 563
one_pulse = 563
one_space = 1688
bit_count = 32

read_mode2_output(header_pulse, header_space, zero_pulse, zero_space, one_pulse, one_space, bit_count)
