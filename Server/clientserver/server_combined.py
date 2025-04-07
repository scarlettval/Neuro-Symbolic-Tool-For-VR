import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 65434))
server_socket.listen()

print("Server is listening on port 65434...")

conn, addr = server_socket.accept()
print(f"Connected by {addr}")

# Receive the message (ends with newline)
msg = b""
while not msg.endswith(b"\n"):
    msg += conn.recv(1)
print("Received message:", msg.decode().strip())

# Receive the file header
header = b""
while not header.endswith(b"\n"):
    header += conn.recv(1)
header = header.decode().strip()
filename, filesize = header.split(";")
filesize = int(filesize)
print(f"Expecting file '{filename}' ({filesize} bytes)")

# Receive the file data
received = 0
with open(f"received_{filename}", "wb") as f:
    while received < filesize:
        data = conn.recv(1024)
        if not data:
            break
        f.write(data)
        received += len(data)

print(f"File '{filename}' received and saved as 'received_{filename}'")
conn.close()
server_socket.close()
