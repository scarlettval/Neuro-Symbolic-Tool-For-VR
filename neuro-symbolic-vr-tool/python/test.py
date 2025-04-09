import sys
import os

# Add the root directory to sys.path so Python can find symbolic_module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from symbolic_module.prolog_interface import send_to_prolog
from mask_rcnn_module import MaskRCNNDetector


def map_label_to_symbolic(label):
    if label == "person":
        return "create(cube)"
    elif label == "bottle":
        return "create(cylinder)"
    elif label == "chair":
        return "create(sphere)"
    elif label == "tv":
        return "delete(cube)"
    elif label == "book":
        return "move(sphere, [1, 0, 0])"
    else:
        return None


detector = MaskRCNNDetector()
image_path = "../images/hand.jpg"


try:
    results = detector.predict(image_path)

    if not results:
        print("No high-confidence objects detected.")
    else:
        for obj in results:
            print(f"Detected: {obj['label']} (score: {obj['score']}) at {obj['box']}")
            symbolic_command = map_label_to_symbolic(obj['label']).rstrip(".")
            if symbolic_command:
                print(f"→ Mapped to symbolic: {symbolic_command}")
                send_to_prolog(symbolic_command)
            else:
                print("→ No symbolic action mapped.")
except FileNotFoundError as e:
    print(e)
