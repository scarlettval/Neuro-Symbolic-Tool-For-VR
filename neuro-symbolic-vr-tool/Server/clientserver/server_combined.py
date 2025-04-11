import socket
import os
import sys

# 🔧 Add the ../python folder to import your modules
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.abspath(os.path.join(current_dir, "../../python"))
sys.path.append(python_dir)

from inference_detectron import run_inference
from symbolic_driver import run_prolog_logic

# Start server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 65434))
server_socket.listen()

print("🟢 Server is listening on port 65434...")

conn, addr = server_socket.accept()
print(f"🔌 Connected by {addr}")

# Receive message
msg = b""
while not msg.endswith(b"\n"):
    msg += conn.recv(1)
print("💬 Received message:", msg.decode().strip())

# Receive header
header = b""
while not header.endswith(b"\n"):
    header += conn.recv(1)
header = header.decode().strip()
filename, filesize = header.split(";")
filesize = int(filesize)
print(f"📦 Receiving file '{filename}' ({filesize} bytes)...")

# Receive file data
received = 0
received_path = f"received_{filename}"
with open(received_path, "wb") as f:
    while received < filesize:
        data = conn.recv(1024)
        if not data:
            break
        f.write(data)
        received += len(data)

print(f"✅ File saved as '{received_path}'")

# Run object detection
print("🧠 Running inference...")
run_inference(received_path)
print("✅ Inference complete")

# Run symbolic reasoning
base = os.path.splitext(filename)[0]
scene_file = f"exported/{base}_scene.json"
if os.path.exists(scene_file):
    print("🔍 Running symbolic reasoning...")
    result = run_prolog_logic(scene_file)
    print("🎯 Symbolic result:", result)
else:
    print("⚠️ scene.json not found. Skipping symbolic reasoning.")

# Close server
conn.close()
server_socket.close()
print("🔚 Server closed.")
