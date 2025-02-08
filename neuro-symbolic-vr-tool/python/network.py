import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure trained_models directory exists
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load and Preprocess Data
df = pd.read_csv("vr_training_data.csv")

# 🔧 Convert categorical labels to numbers
posture_mapping = {"standing": 0, "crouching": 1, "jumping": 2}
df["Posture"] = df["Posture"].map(posture_mapping)

# 🔧 Ensure all columns exist in the dataset
required_columns = ["HandTracking", "Controller", "ObjectNearby", "Posture", "HandX", "HandY", "HandZ", "Action"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# Convert dataset to numpy array and cast to float32
X_train = df[["HandTracking", "Controller", "ObjectNearby", "Posture", "HandX", "HandY", "HandZ"]].values.astype(np.float32)
y_train = df["Action"].values

# 🔧 Convert labels to numerical values (e.g., "move_forward" -> 0)
unique_actions = list(set(y_train))
action_mapping = {action: idx for idx, action in enumerate(unique_actions)}
y_train = np.array([action_mapping[action] for action in y_train], dtype=np.int64)

# ✅ Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# ✅ Define Neural Network Model
# ✅ Define VRActionModel if missing
class VRActionModel(nn.Module):
    def __init__(self, input_size=7, output_size=3):  # ✅ Match saved model
        super(VRActionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # ✅ Ensure correct input size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)  # ✅ Ensure correct output size

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ✅ Initialize Model
input_size = X_train.shape[1]  # Assuming the second dimension is features
output_size = len(unique_actions)
model = VRActionModel(input_size, output_size)

# ✅ Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 🚀 Train Model
print("🚀 Training with VR data...")
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ✅ Save trained model
model_path = os.path.join(MODEL_DIR, "vr_model.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved successfully at: {model_path}")

# ✅ Save action mapping for future decoding
mapping_path = os.path.join(MODEL_DIR, "action_mapping.npy")
np.save(mapping_path, action_mapping)
print(f"✅ Action mapping saved at: {mapping_path}")
