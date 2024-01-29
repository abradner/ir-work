import socket
import time
import random

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the server's address and port
server_address = ('localhost', 12345)
sock.connect(server_address)

try:
    # Send data
    for _ in range(1000):
      # Generate some sample data
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        data = f"{x},{y}\n"

        # Send the data
        sock.sendall(data.encode('utf-8'))

        # Wait a bit bmv grefore sending the next batch of data
        time.sleep(0.001)

    print("Finished sending data.")
    # Wait for 3 seconds before disconnecting
    time.sleep(3)

finally:
    # Clean up the connection
    sock.close()
