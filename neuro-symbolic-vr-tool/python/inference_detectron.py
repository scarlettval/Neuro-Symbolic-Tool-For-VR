import os
import cv2
import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("output/clevr_config.yaml")
    cfg.MODEL.WEIGHTS = "output/clevr_model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = "cpu"
    return cfg


def load_class_labels():
    with open("output/clevr_labels.json", "r") as f:
        return json.load(f)


def run_inference(image_path, predictor, thing_classes):
    image = cv2.imread(image_path)
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    # Inject class labels
    MetadataCatalog.get("clevr_train").thing_classes = thing_classes

    # Draw visualization
    visualizer = Visualizer(image[:, :, ::-1], MetadataCatalog.get("clevr_train"), scale=1.2)
    out = visualizer.draw_instance_predictions(instances)

    # Save visual result
    os.makedirs("output", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    visual_output_path = f"output/{base_name}_predicted.png"
    cv2.imwrite(visual_output_path, out.get_image()[:, :, ::-1])
    print(f"✅ Saved image with predictions: {visual_output_path}")

    # Extract prediction info
    scene_data = []
    for box, score, cls in zip(instances.pred_boxes.tensor.tolist(), instances.scores.tolist(), instances.pred_classes.tolist()):
        scene_data.append({
            "label": thing_classes[cls] if cls < len(thing_classes) else f"class_{cls}",
            "bbox": box,
            "score": round(score, 4)
        })

    # Save scene.json
    os.makedirs("exported", exist_ok=True)
    scene_output_path = f"exported/{base_name}_scene.json"
    with open(scene_output_path, "w") as f:
        json.dump(scene_data, f, indent=2)
    print(f"📄 Saved scene data: {scene_output_path}")


if __name__ == "__main__":
    # Prepare config and model
    print("🔧 Setting up model...")
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    thing_classes = load_class_labels()

    # Folder with validation images
    image_dir = "data/CLEVR_v1.0/images/val"
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    print(f"\n🧠 Running inference on {len(image_files)} image(s)...\n")
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        print(f"▶️ [{idx+1}/{len(image_files)}] Processing {image_file}")
        run_inference(image_path, predictor, thing_classes)
