import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ✅ Ensure trained_models directory exists
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load Dataset
csv_path = "vr_training_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ Training data not found at {csv_path}")

df = pd.read_csv(csv_path)

# ✅ Check for missing values
if df.isna().sum().sum() > 0:
    raise ValueError("❌ Dataset contains missing values! Please clean `vr_training_data.csv`.")

# ✅ Encode Categorical Data
posture_mapping = {"standing": 0, "crouching": 1, "jumping": 2}
df["Posture"] = df["Posture"].map(posture_mapping)

# ✅ Ensure required columns exist
required_columns = ["HandTracking", "Controller", "ObjectNearby", "Posture", "HandX", "HandY", "HandZ", "Action"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns in dataset: {missing_cols}")

# ✅ Extract Features and Labels
X_train = df[["HandTracking", "Controller", "ObjectNearby", "Posture", "HandX", "HandY", "HandZ"]].values.astype(np.float32)
y_train = df["Action"].values

# ✅ Convert actions to numeric encoding
unique_actions = sorted(set(y_train))
action_mapping = {action: idx for idx, action in enumerate(unique_actions)}
y_train = np.array([action_mapping[action] for action in y_train], dtype=np.int64)

# ✅ Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# ✅ Define Neural Network Model
class VRActionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(VRActionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ✅ Initialize Model
input_size = X_train.shape[1]  # Auto-detect number of features
output_size = len(unique_actions)  # Auto-detect number of actions
model = VRActionModel(input_size, output_size)

# ✅ Apply Weight Initialization (Fixes training instability)
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.apply(initialize_weights)

# ✅ Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate

# 🚀 Train Model
print("🚀 Training VR Action Model...")
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    if torch.isnan(loss):
        raise ValueError("❌ Loss became NaN! Check dataset and training stability.")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevents exploding gradients
    optimizer.step()

    # Log every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ✅ Save trained model
model_path = os.path.join(MODEL_DIR, "vr_model.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved successfully at: {model_path}")

# ✅ Save action mapping for future decoding
mapping_path = os.path.join(MODEL_DIR, "action_mapping.npy")
np.save(mapping_path, action_mapping)
print(f"✅ Action mapping saved at: {mapping_path}")
