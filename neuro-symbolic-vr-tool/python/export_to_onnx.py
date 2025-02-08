import torch
import torch.nn as nn
import os

# ✅ Ensure the model definition matches the trained model in network.py
class VRActionModel(nn.Module):
    def __init__(self, input_size=7, output_size=3):  # ✅ Match these values to network.py
        super(VRActionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ✅ Ensure the trained model path is correct
model_dir = "trained_models"
model_path = os.path.join(model_dir, "vr_model.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}. Train the model first.")

# ✅ Load the trained model
input_size = 7  # ✅ Ensure this matches network.py
output_size = 3  # ✅ Ensure this matches network.py
model = VRActionModel(input_size, output_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# ✅ Create dummy input for ONNX export
dummy_input = torch.randn(1, input_size)  # Match input size

# ✅ Set ONNX export path
onnx_path = os.path.join(model_dir, "vr_model.onnx")

# ✅ Export the model
torch.onnx.export(
    model, dummy_input, onnx_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11  # Ensure compatibility with Unity Barracuda
)

print(f"✅ Model successfully exported to {onnx_path}")
