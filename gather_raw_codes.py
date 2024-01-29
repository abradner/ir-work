import subprocess

class RawCodeGatherer:
    def __init__(self):
        self.process = self._new_process()

    def gather_signal(self, message_length):
        raw_signal = []
        while True:
            part = self._fetch_code_part()
            if not part:
                print("WARN: Could not read line from mode2 output.")
                break
            raw_signal.append(part)
            if len(raw_signal) >= message_length:
                break
            elif part == 'timeout':
                break

        # print(f"Read raw signal: {raw_signal}")
        return raw_signal

    def gather_pair(self):
        raw_signal = self.gather_signal(2)
        if len(raw_signal) < 2:
            print ("WARN: Could not read pair from mode2 output.")
            return None
        return [raw_signal[0][1], raw_signal[1][1]]

    def close(self):
        self.process.kill()

    def _new_process(self):
        return subprocess.Popen(['mode2'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _fetch_code_part(self):
        """Read and decode a line of output from mode2 command."""
        line = self._read_line()
        if not line:
            return None
        return self._interpret_line(line)

    def _read_line(self):
        """Read a line of output from mode2 command."""
        raw_line = self.process.stdout.readline()
        if not raw_line:
            return None
        return raw_line.decode('utf-8').strip()

    def _interpret_line(self, line):
        """Decode a line from the process output."""
        parts = line.split()
        if len(parts) == 2 and parts[0] in ['pulse', 'space', 'timeout']:
            signal_type, value = parts
            return signal_type, int(value)
        return None
