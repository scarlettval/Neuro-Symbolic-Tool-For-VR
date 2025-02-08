import onnxruntime as ort
import numpy as np

# Load the ONNX model
onnx_path = "trained_models/vr_model.onnx"
session = ort.InferenceSession(onnx_path)

# Create a dummy input
test_input = np.random.randn(1, 7).astype(np.float32)  # Match input size

# Run inference
outputs = session.run(None, {"input": test_input})
print("ONNX Model Output:", outputs)
