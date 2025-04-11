import socket
import os
import sys
import time

# === CONFIGURATION ===

# 1. Fallback screenshot filename for manual testing
DEFAULT_FILENAME = "screenshot.png"

# 2. Directory where VR screenshots are stored (update this when using Meta Quest/Unity)
VR_SCREENSHOT_DIR = "."  # Put actual VR screenshot path here later


# === FIND THE MOST RECENT SCREENSHOT ===

def find_latest_screenshot(directory):
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        return None
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, latest_file)


# === MAIN CLIENT FUNCTION ===

def send_file_to_server(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File '{filepath}' not found.")

    filename = os.path.basename(filepath)
    filesize = os.path.getsize(filepath)

    print(f"📤 Sending file '{filename}' ({filesize} bytes) to server...")

    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 65434))

    # Send message
    message = f"Sending VR screenshot: {filename}"
    client_socket.sendall(message.encode() + b"\n")

    # Send header
    header = f"{filename};{filesize}"
    client_socket.sendall(header.encode() + b"\n")

    # Send file data
    with open(filepath, "rb") as f:
        while chunk := f.read(1024):
            client_socket.sendall(chunk)

    print("✅ File sent successfully!")
    client_socket.close()


# === ENTRY POINT ===

if __name__ == "__main__":
    # Optional override: allow CLI filename
    if len(sys.argv) > 1:
        selected_file = sys.argv[1]
    else:
        # Find the latest image in screenshot directory
        selected_file = find_latest_screenshot(VR_SCREENSHOT_DIR)

    if selected_file is None:
        print("❌ No screenshots found to send.")
    else:
        send_file_to_server(selected_file)
