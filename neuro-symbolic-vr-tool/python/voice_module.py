import speech_recognition as sr

def get_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening for voice command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print("🗣️ Recognized:", command)
        return command.lower()
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"❌ Speech recognition error: {e}")
        return None

def parse_command_to_symbolic(text):
    if not text:
        return None

    text = text.lower().strip()

    # Normalize phrasing
    text = text.replace(" to the ", " ")
    words = text.split()

    # Try to extract relevant parts after "move"
    if "move" not in words:
        return None

    try:
        idx = words.index("move")

        # Defensive defaults
        size = color = shape = direction = ""

        # Look ahead safely
        for i in range(idx + 1, len(words)):
            word = words[i]
            if word in {"small", "large"}:
                size = word
            elif word in {
                "red", "blue", "green", "yellow", "cyan", "purple", "gray", "brown"
            }:
                color = word
            elif word in {"cube", "sphere", "cylinder"}:
                shape = word
            elif word in {"left", "right"}:
                direction = word

        if not (size and color and shape and direction):
            print("⚠️ Incomplete voice command.")
            return None

        return f"move_the_{size}_{color}_{shape}_to_the_{direction}"
    except Exception as e:
        print(f"⚠️ Failed to parse command: {e}")
        return None
