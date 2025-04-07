import socket
import os

# Connect to server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 65434))  # New port to avoid conflicts

# Send message
message = "Hello from the client! Here’s an image file coming your way!"
client_socket.sendall(message.encode() + b"\n")  # Send message with newline as delimiter

# Send image file
filename = "s.jpeg"
filesize = os.path.getsize(filename)
header = f"{filename};{filesize}"
client_socket.sendall(header.encode() + b"\n")  # Header with newline

# Send file data
with open(filename, "rb") as f:
    while chunk := f.read(1024):
        client_socket.sendall(chunk)

print("Message and file sent!")
client_socket.close()
