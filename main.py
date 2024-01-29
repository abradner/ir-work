#! /bin/env python3

import json
import os
import sys
from tuning import tune_ir_structure
from gather_raw_codes import RawCodeGatherer
from signal_processing import process_signal

def main():

    message_length = 32 # Number of bits in the message
    output_file = sys.argv[1]  # Get output file from command line args
    gatherer = RawCodeGatherer()

    # Load tuning data if output file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            tuning_data = json.load(f)
    else:
        tuning_data = None


    # trigger tuning if tuning data is not available
    if not tuning_data:
        print("Tuning IR signal structure...")
        tuning_data = tune_ir_structure(gatherer)
        print(f"Tuned header_pulse: {tuning_data['header_pulse']}, header_space: {tuning_data['header_space']}, zero_pulse: {tuning_data['zero_pulse']}, zero_space: {tuning_data['zero_space']}, one_pulse: {tuning_data['one_pulse']}, one_space: {tuning_data['one_space']}, tolerance: {tuning_data['tolerance']}")


    busy_loop(message_length, tuning_data, output_file, gatherer)


def busy_loop(message_length, tuning_data, output_file, gatherer):
    """Busy loop to gather and process IR signals."""

    print("Reading IR signal...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            signal = gatherer.gather_signal(message_length)
            process_signal(signal, tuning_data)

    except KeyboardInterrupt:
        # Save tuning data to output file
        with open(output_file, 'w') as f:
            json.dump(tuning_data, f)
        gatherer.close()


if __name__ == '__main__':
    main()
