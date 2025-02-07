import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# ✅ Load VR Training Data
df = pd.read_csv("vr_training_data.csv")

# ✅ Convert Categorical Data (Posture) to Numbers
posture_mapping = {"standing": 0, "crouching": 1, "jumping": 2}
df["Posture"] = df["Posture"].map(posture_mapping)

# ✅ Prepare Inputs & Labels
X_train = df[["HandTracking", "Controller", "ObjectNearby", "Posture", "HandX", "HandY", "HandZ"]].values
y_train = df["Action"].astype("category").cat.codes.values
action_labels = df["Action"].astype("category").cat.categories.tolist()

# ✅ Convert to Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# ✅ Define Improved Neural Network
class VRActionPredictor(nn.Module):
    def __init__(self):
        super(VRActionPredictor, self).__init__()
        self.fc1 = nn.Linear(7, 10)  # More input features
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, len(action_labels))  # Dynamic action count
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# ✅ Train the Model
model = VRActionPredictor()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("🚀 Training with VR data...")
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}")

# ✅ Save Model
torch.save(model.state_dict(), "vr_action_model.pth")
print("✅ Model training complete with real VR data!")

# ✅ Test Prediction
example_input = torch.tensor([[1, 1, 1, 0, 0.5, 0.6, 0.7]])  # Example scenario
predicted_action = action_labels[torch.argmax(model(example_input)).item()]
print(f"🤖 Predicted Action: {predicted_action}")
