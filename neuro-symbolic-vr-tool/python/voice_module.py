import speech_recognition as sr
from datetime import datetime

def get_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(f" Listening for voice command... ({datetime.now()})")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f" Recognized: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print(" Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f" Speech recognition error: {e}")
        return ""

# Optional legacy parser, no longer needed if you're using dynamic interpret/2
def parse_command_to_symbolic(text):
    return None  # You now use the raw string directly in Prolog
