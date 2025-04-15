import socket
import json

def send_action_to_unity(json_path="output/symbolic_action.json", host="127.0.0.1", port=5050):
    with open(json_path, "r") as f:
        data = json.load(f)

    payload = json.dumps(data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(payload.encode('utf-8'))
        print("✅ Sent to Unity:", payload)
