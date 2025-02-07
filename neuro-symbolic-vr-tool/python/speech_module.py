import speech_recognition as sr

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

if __name__ == "__main__":
    command = recognize_speech()
    if command:
        print(f"✅ Recognized Command: {command}")
    else:
        print("⚠️ No valid speech recognized.")
