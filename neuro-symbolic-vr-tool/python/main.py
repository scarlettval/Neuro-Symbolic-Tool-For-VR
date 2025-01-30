import torch
from speech_module import recognize_speech
from network import SimpleNN

device = torch.device("cpu")

def main():
    print("🎙️ Starting Speech Recognition...")
    recognized_text = recognize_speech()  
    
    if recognized_text:
        print(f"✅ Recognized Speech: {recognized_text}")

    print("\n🤖 Running Neural Network...")
    model = SimpleNN().to(device)
    
    
    sample_input = torch.tensor([[1.0, 2.0]]).to(device)
    
    output = model(sample_input)
    print(f"✅ Neural Network Output: {output.item()}")

if __name__ == "__main__":
    main()
