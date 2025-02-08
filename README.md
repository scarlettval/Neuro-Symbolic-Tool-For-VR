# Neuro-Symbolic VR Tool

A cutting-edge **Virtual Reality** system that integrates **Neural Networks** and **Symbolic Reasoning** to create an intelligent, immersive experience on the **Meta Quest 2**. This tool leverages state-of-the-art **machine learning**, **Prolog-based symbolic logic**, and **speech recognition** to deliver adaptive, intuitive VR interactions. By combining learning-based approaches with explicit logic rules, it enables more transparent decision-making, real-time adaptability, and deeper user engagement.

---

## Table of Contents
1. [Features](#features)  
2. [Architecture Overview](#architecture-overview)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Development Roadmap](#development-roadmap)  
7. [Contributing](#contributing)  
8. [License](#license)  
9. [Acknowledgments](#acknowledgments)

---

## Features

- **Immersive VR**: Built for the **Meta Quest 2** using Unity’s XR toolkit (or Oculus Integration).  
- **Neural Network Module**: Implemented in Python (e.g., PyTorch), enabling:
  - Real-time **inference** and **dynamic retraining**.
  - Seamless data exchange with symbolic logic.
- **Symbolic Reasoning Module**: Uses **Prolog** (via [pyswip](https://github.com/yuce/pyswip)) to:
  - Embed logic rules that handle explicit knowledge and constraints.
  - Complement the neural network with explainable decision paths.
- **Speech Recognition**: Integrate with:
  - [PocketSphinx](https://github.com/cmusphinx/pocketsphinx) or
  - [OpenAI Whisper](https://github.com/openai/whisper)
  - Extensible for other APIs (Google, Azure, etc.).
- **Intuitive UI**: VR-based user interface for effortless interactions and fast feedback loops.

---
# INSTRUCTION FOR GROUP MEMEBERS


 Setup Instructions
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/hanene152/Symbolic-Tool-For-Virtual-Reality.git
cd Symbolic-Tool-For-Virtual-Reality/neuro-symbolic-vr-tool
2️⃣ Create a Virtual Environment (Optional but Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows (PowerShell)
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Verify Installation
bash
Copy
Edit
python -c "import torch, onnx, speech_recognition, logicpy, pandas, numpy; print('✅ All libraries loaded successfully!')"
🔧 Running the Project
1️⃣ Train the ML Model
bash
Copy
Edit
python python/network.py
Trains the model on vr_training_data.csv
Saves the trained model in trained_models/vr_model.pth
2️⃣ Convert Model to ONNX for Unity
bash
Copy
Edit
python python/export_to_onnx.py
Converts vr_model.pth to vr_model.onnx (saved in trained_models/)
3️⃣ Run Symbolic Reasoning
bash
Copy
Edit
python python/prolog_interface.py
Uses symbolic logic to determine valid actions based on current conditions.
4️⃣ Test Speech Recognition
bash
Copy
Edit
python python/speech_module.py
Uses speech_recognition to interpret voice commands.