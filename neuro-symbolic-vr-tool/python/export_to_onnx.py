import torch
import torch.nn as nn
import torch.onnx
import os

# Define the same model structure as used in training
class VRActionModel(nn.Module):
    def __init__(self):
        super(VRActionModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)  # Example input features (adjust based on real VR input)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 6)  # Example output actions (6 possible VR actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load trained model (update path to match your trained model)
model_path = "trained_models/vr_model.pth"
model = VRActionModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Define dummy input (should match the real input shape)
dummy_input = torch.randn(1, 3)  # Example: 3 input features

# Set the ONNX output path (ensure Unity has access)
onnx_model_path = "VRActionModel.onnx"
output_dir = "Assets/Models/"  # Unity project path (adjust as needed)
os.makedirs(output_dir, exist_ok=True)
full_onnx_path = os.path.join(output_dir, onnx_model_path)

# Export model to ONNX
torch.onnx.export(
    model, dummy_input, full_onnx_path,
    export_params=True,  
    opset_version=11,
    do_constant_folding=True,  
    input_names=["input"],  
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"✅ ONNX Model exported successfully to: {full_onnx_path}")
