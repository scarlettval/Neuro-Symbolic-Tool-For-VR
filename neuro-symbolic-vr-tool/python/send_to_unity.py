import json
import socket

def send_action_to_unity(json_path="output/symbolic_action.json"):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        host = "localhost"
        port = 5050
        json_msg = json.dumps(data)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(json_msg.encode("utf-8"))

        print(f"✅ Sent to Unity: {json_msg}")

    except FileNotFoundError:
        print("❌ symbolic_action.json not found.")
    except ConnectionRefusedError:
        print("❌ Unity server is not running.")
    except Exception as e:
        print(f"❌ Error sending to Unity: {e}")
