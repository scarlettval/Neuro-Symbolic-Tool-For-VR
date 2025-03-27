import sys
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# Add module paths
sys.path.append(os.path.abspath("python"))
sys.path.append(os.path.abspath("symbolic_module"))

from speech_module import recognize_speech
from mask_rcnn_module import MaskRCNNDetector
from symbolic_logger import save_symbolic_command
from prolog_interface import send_to_prolog

def map_label_to_symbolic(label):
    if label == "person":
        return "create(cube)"
    elif label == "bottle":
        return "create(cylinder)"
    elif label == "chair":
        return "create(sphere)"
    return None

def map_combined_input(speech, vision_label):
    speech = speech.lower()
    if "create" in speech and "cube" in speech:
        return "create(cube)"
    elif "delete" in speech and "sphere" in speech:
        return "delete(sphere)"
    elif "move" in speech and "cylinder" in speech:
        return "move(cylinder, [1, 0, 0])"
    return map_label_to_symbolic(vision_label)

def visualize_results(image_path, results):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for obj in results:
        box = obj["box"]
        label = obj["label"]
        score = obj["score"]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 5, box[1] + 5), f"{label} ({score})", fill="red")

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Vision Detection Output")
    plt.show()

def run_pipeline_test():
    print("🧠 Neuro-Symbolic Pipeline Test (Speech + Vision + Prolog)\n")

    # Step 1: Speech
    print("🎙️ Say a command... (Listening)")
    spoken_text = recognize_speech()
    print(f"✅ Recognized Command: {spoken_text}")

    # Step 2: Visual Detection
    detector = MaskRCNNDetector()
    image_path = os.path.join("images", "hand.jpg")
    results = detector.predict(image_path)

    if not results:
        print("🚫 No objects detected.")
        return

    visualize_results(image_path, results)

    for obj in results:
        label = obj["label"]
        print(f"🖼️ Detected: {label} (score: {obj['score']})")

        symbolic_command = map_combined_input(spoken_text, label)
        if symbolic_command:
            print(f"🔁 Mapped Symbolic Command: {symbolic_command}")
            send_to_prolog(symbolic_command.rstrip("."))
        else:
            print("⚠️ No symbolic mapping found.")

if __name__ == "__main__":
    run_pipeline_test()
