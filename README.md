Markdown
# 🌾 HarvestIQ — AI Agro Diagnostic Suite (v2.0 Universal)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-2.0%20Universal-22c55e.svg)](#)
[![Built With](https://img.shields.io/badge/Built%20With-FastAPI%20%7C%20Flask%20%2B%20TailwindCSS-0d1520.svg)](#)

**HarvestIQ** is an end-to-end intelligent agricultural diagnosis and decision-support workspace. Powered by deep neural vision models and localized LLM reasoning, HarvestIQ enables farmers, agronomists, and researchers to detect crop leaf pathologies, review localized agronomic advisories, monitor live microclimate conditions, and interact via voice-enabled multilingual conversational AI.

---

## ✨ Features

- **Multi-Crop Pathology Scanner:** Real-time visual diagnosis for core staple crops: **Rice, Corn, Wheat, and Potato**.
- **Visual Attention/Explainability:** Dynamic rendering of diagnostic overlays (`attention_overlay`) highlighting detected lesions and infected zones.
- **Automated Agronomic Prescription:** Markdown-formatted diagnostic reports detailing pathology names, model confidence scores, and targeted treatment advisories.
- **HarvestBot Agronomic Assistant:** Context-aware chatbot trained on soil chemistry, pest management, and disease mitigation strategies.
- **Weather & Microclimate Integration:** Automatic geolocation-based atmospheric sensor sync (temperature, relative humidity) via Open-Meteo API.
- **Multilingual Support & Voice Input:** Native speech-to-text recognition and multilingual text generation in **English**, **Hindi (हिंदी)**, and **Bengali (বাংলা)**.
- **One-Click Diagnostic Export:** Client-side generation and export of formal PDF prescription sheets via `html2pdf.js`.
- **Modern Dark-Mode UI:** Responsive glassmorphic layout styled with Tailwind CSS, custom glow effects, and interactive feedback states.

---

## 🛠️ Architecture & Tech Stack

### Client-Side (Frontend)
- **HTML5 & Vanilla JavaScript:** Dynamic state handling, Web Speech API integration, and asynchronous backend communication.
- **Tailwind CSS:** Modern utility-first dark-mode UI with custom palettes (`harvestDark`, `harvestGreen`, `harvestGold`).
- **Marked.js:** Dynamic client-side parsing of Markdown prescription advisories and chat streams.
- **html2pdf.js:** High-resolution DOM-to-PDF export pipeline.
- **Open-Meteo API:** Real-time sensor and weather parameter ingestion.

### Server-Side (Backend Blueprint)
- **Framework:** Python (FastAPI / Flask)
- **Vision Engine:** Deep Learning / Computer Vision model (e.g., PyTorch, TensorFlow/Keras, ONNX Runtime) trained on agricultural disease datasets (e.g., PlantVillage).
- **Advisory Engine:** Local or Cloud LLM API configured for agronomic reasoning and multilingual translation.

---

## 📁 Suggested Repository Structure

```text
HarvestIQ/
├── backend/
│   ├── app.py                # Main server (FastAPI / Flask)
│   ├── model/
│   │   ├── classifier.onnx   # Trained crop pathology model
│   │   └── labels.json       # Crop and disease class indices
│   ├── services/
│   │   ├── vision.py         # Image inference & Grad-CAM pipeline
│   │   └── chat.py           # LLM agent logic & prompt engineering
│   └── requirements.txt      # Python dependencies
├── frontend/
│   └── index.html            # HarvestIQ UI (Single-Page Application)
├── assets/                   # Architecture diagrams & screenshots
├── .gitignore
├── LICENSE
└── README.md
🔌 API Contract Reference
The frontend expects the backend server to expose the following endpoints:

1. Leaf Diagnosis Endpoint
URL: /analyze

Method: POST

Content-Type: multipart/form-data

Request Parameters:

file: Image file (leaf image)

language: English | Hindi | Bengali

weather: String formatted microclimate data (e.g., "28°C, 75% Humidity")

Response Format:

JSON
{
  "crop": "Potato",
  "disease": "Early Blight (Alternaria solani)",
  "confidence": 0.942,
  "is_healthy": false,
  "remedy": "### Recommended Interventions\n* Apply **Mancozeb 75% WP** at 2g/L.\n* Improve air circulation and eliminate furrow water stagnation.",
  "attention_overlay": "data:image/jpeg;base64,...",
  "error": null
}
2. Conversational Assistant Endpoint
URL: /chat

Method: POST

Content-Type: application/json

Request Body:

JSON
{
  "message": "What fungicide is best for potato early blight under high humidity?",
  "language": "English",
  "weather": "28°C, 75% Humidity"
}
Response Format:

JSON
{
  "response": "Under high humidity (75%), fungal spread accelerates. Prioritize systemic fungicides containing **Chlorothalonil** or **Azoxystrobin**..."
}
🚀 Quick Start
1. Clone the Repository
Bash
git clone [https://github.com/your-username/HarvestIQ.git](https://github.com/your-username/HarvestIQ.git)
cd HarvestIQ
2. Set Up the Backend
Using Python 3.9+:

Bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pillow torch torchvision  # Example stack
3. Run the Development Server
Bash
# Assuming backend/app.py serves static files or runs on port 8000
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
Open your browser and navigate to http://localhost:8000 to launch the workspace.

🧪 Browser Compatibility
Feature	Chrome / Edge	Firefox	Safari
Glassmorphic UI & Layout	✅	✅	✅
Geolocation Sensor Fetch	✅	✅	✅
PDF Prescription Export	✅	✅	✅
Voice Recognition (Web Speech API)	✅	⚠️ (Flag-dependent)	⚠️ (Partial)
👨‍💻 Author & Engineering Credits
Engineered with precision by Team CodoLeoX16.

For inquiries, collaborations, or model fine-tuning discussions, open an issue or pull request in this repository.

📄 License
This project is licensed under the MIT License — feel free to modify and adapt it for agricultural research and production deployments.
