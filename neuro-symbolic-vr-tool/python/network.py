import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ✅ Define Training Data (Features: Hand Tracking, Controller, Object Nearby)
X_train = np.array([
    [1, 0, 1],  # Hand tracking, no controller, object nearby -> grab_object
    [0, 1, 0],  # No hand tracking, controller enabled, no object -> move_forward
    [1, 0, 0],  # Hand tracking, no controller, no object -> wave_hand
    [0, 1, 1],  # No hand tracking, controller, object nearby -> press_button
    [1, 1, 1],  # Both enabled, object nearby -> grab_object
    [0, 1, 0],  # Controller only -> move_forward
    [1, 0, 0],  # Hand tracking only -> wave_hand
    [0, 1, 1],  # Controller, object nearby -> press_button
])

# ✅ Corresponding Actions (Labels)
y_train = np.array([
    0,  # grab_object
    1,  # move_forward
    2,  # wave_hand
    3,  # press_button
    0,  # grab_object
    1,  # move_forward
    2,  # wave_hand
    3,  # press_button
])

# ✅ Convert to Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# ✅ Define Action Labels
action_labels = ["grab_object", "move_forward", "wave_hand", "press_button"]

# ✅ Define Improved Neural Network
class VRActionPredictor(nn.Module):
    def __init__(self):
        super(VRActionPredictor, self).__init__()
        self.fc1 = nn.Linear(3, 8)  # More neurons for better feature learning
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 4)  # 4 possible actions
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# ✅ Train the Model
model = VRActionPredictor()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("🚀 Training model...")
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
print("✅ Model training complete and saved!")

# ✅ Test Prediction
example_input = torch.tensor([[1.0, 0.0, 1.0]])  # Hand tracking, object nearby
predicted_action = action_labels[torch.argmax(model(example_input)).item()]
print(f"🤖 Predicted Action: {predicted_action}")
