from pyDatalog import pyDatalog
import speech_recognition as sr
import torch
import torch.nn as nn

# ✅ Define symbolic logic terms
pyDatalog.create_terms('action, condition, effect, interactable, object, X, Y, Z, exists, size, position, rotation')

# ✅ Object Creation Rules
+action("spawn", "can_spawn")  # Can create an object if spawning is enabled
+action("delete", "exists")  # Can delete an object if it exists

# ✅ Object Movement Rules
+action("move", "exists")  # Can move objects if they exist
+action("resize", "exists")  # Can resize objects if they exist
+action("rotate", "exists")  # Can rotate objects if they exist

# ✅ Interaction Rules (Hand tracking removed)
+interactable("cube")
+interactable("sphere")
+interactable("pyramid")
+interactable("cylinder")

# ✅ Default World Conditions
+condition("can_spawn")  # Users are allowed to spawn objects
+condition("exists") <= object(X)  # Objects must exist to interact with them

effect(X) <= action(X, Y) & condition(Y)

# ✅ Default Object Properties (Assign initial values)
+object("cube")  # Example object
+size("cube", 1)  # Default size
+position("cube", (0, 0, 0))  # Default position
+rotation("cube", (0, 0, 0))  # Default rotation

# ✅ Define symbolic logic terms
pyDatalog.create_terms('action, condition, effect, interactable, required, X, Y')

# ✅ Expanded Symbolic VR Actions
+action("grab_object", "hand_tracking_enabled")
+action("throw_object", "hand_tracking_enabled")
+action("push_object", "hand_tracking_enabled")
+action("press_button", "controller_enabled")
+action("open_door", "controller_enabled")
+action("wave_hand", "hand_tracking_enabled")
+action("point", "hand_tracking_enabled")
+action("jump", "controller_enabled")
+action("crouch", "controller_enabled")
+action("move_forward", "controller_enabled")
+action("move_backward", "controller_enabled")
+action("rotate_left", "controller_enabled")
+action("rotate_right", "controller_enabled")

# ✅ Define available conditions
+condition("hand_tracking_enabled")
+condition("controller_enabled")
+condition("object_nearby")

# ✅ Define logic for valid actions
effect(X) <= action(X, Y) & condition(Y)

# ✅ Load ML Model for action prediction
class VRActionPredictor(nn.Module):
    def __init__(self):
        super(VRActionPredictor, self).__init__()
        self.fc1 = nn.Linear(3, 5)  # Example input: Hand Tracking, Controller, Object Nearby
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 6)  # Output: Different VR actions
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

model = VRActionPredictor()

# ✅ Voice Recognition: Capture and process voice command
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎙️ Say a command... (Listening)")
        try:
            audio = recognizer.listen(source, timeout=5)
            recognized_text = recognizer.recognize_google(audio).lower()
            return recognized_text
        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
            return None
        except sr.RequestError:
            print("❌ Could not reach speech recognition service.")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

# ✅ Query symbolic reasoning with voice input
def query_with_voice():
    command = recognize_speech()
    if not command:
        return

    print(f"\n🤖 Recognized Voice Command: {command}")

    # ✅ Normalize the voice command to match symbolic rules
    formatted_command = command.replace(" ", "_")

    # ✅ ML Predicts an Action
    example_input = torch.tensor([[1.0, 0.0, 1.0]])  # Example input
    with torch.no_grad():
        prediction = model(example_input)

    actions = ["grab_object", "move_forward", "wave_hand", "press_button", "jump", "crouch"]
    predicted_action = actions[torch.argmax(prediction).item()]

    print(f"🤖 ML Predicted Action: {predicted_action}")

    # ✅ Validate with Symbolic Reasoning
    valid_actions = [x[0] for x in effect(X)]  # Extracts valid actions

    if formatted_command in valid_actions:
        print(f"✅ Action {formatted_command} is allowed!")
    else:
        print(f"❌ Action {formatted_command} is NOT allowed! Suggesting alternatives...")
        print(f"✅ Allowed Actions: {valid_actions}")


if __name__ == "__main__":
    query_with_voice()