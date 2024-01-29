import queue
import socket
import threading

from time import sleep
from gather_raw_codes import RawCodeGatherer

def gather_raw_button_data(q, signal_count, lock):
  rcg = RawCodeGatherer()

  print("Press a button on the remote control")
  while True:
    pair = rcg.gather_pair()
    q.put(pair)
    with lock:
      signal_count[0] += 1

def transmit_raw_button_data(q, transmit_count, lock):
  server_address = ('192.168.1.22', 12345)
  print('connecting to {} port {}'.format(*server_address))

  # Create a TCP/IP socket
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  # Connect the socket to the server's address and port
  sock.connect(server_address)

  try:
    # Send data
    print('connected, ready to send data')  
        
    while True:
      pair = q.get()
      x, y = pair
      data = f"{x},{y}\n"

      # Send the data
      sock.sendall(data.encode('utf-8'))
      with lock:
        transmit_count[0] += 1

  finally:
    # Clean up the connection
    sock.close()

def main():
  q = queue.Queue()
  signal_count = [0]
  transmit_count = [0]

  lock = threading.Lock()

  last_signal_count = 0
  last_transmit_count = 0

  # t1 = threading.Thread(target=gather_raw_button_data, daemon=True, args=(q,signal_count, lock))
  t2 = threading.Thread(target=transmit_raw_button_data, daemon=True, args=(q,transmit_count, lock))

  t2.start()
  # t1.start()

  print("Press Ctrl+C to exit")

  rcg = RawCodeGatherer()
  # gather_raw_button_data(q, signal_count)
  # exit()

  try:
    print("Press a button on the remote control")
    while True:
      pair = rcg.gather_pair()
      q.put(pair)
      with lock:
        signal_count[0] += 1

      if signal_count[0] != last_signal_count or transmit_count[0] != last_transmit_count:
        print(f"signal_count: {signal_count}, transmit_count: {transmit_count}")
        last_signal_count = signal_count
        last_transmit_count = transmit_count

  except KeyboardInterrupt:
      print("Exiting...")
      # t1.join()
      t2.join()

if __name__ == "__main__":
  main()
