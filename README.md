<div align="center">

# 🌾 HarvestIQ
### AI Agro Diagnostic Suite `v2.0 Universal`

An intelligent agricultural diagnostic and conversational advisory system powered by Vision Transformers (ViT), attention rollout explainability, localized weather telemetry, and multi-LLM agronomic reasoning.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Render_Cloud-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://harvestiq-ai-powered-agriculture.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Hugging Face](https://img.shields.io/badge/Vision_Model-ViT_Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/wambugu71/crop_leaf_diseases_vit)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Tailwind CSS](https://img.shields.io/badge/UI-TailwindCSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br />

### 🌐 [Launch Live Web Application →](https://harvestiq-ai-powered-agriculture.onrender.com/)

</div>

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Core AI & ML Architecture](#-core-ai--ml-architecture)
- [Repository Structure](#-repository-structure)
- [Environment Configuration](#-environment-configuration)
- [API Specification](#-api-specification)
  - [1. Diagnostic Vision Engine (`/analyze`)](#1-diagnostic-vision-engine-analyze)
  - [2. Conversational Agronomist (`/chat`)](#2-conversational-agronomist-chat)
- [Local Setup & Installation](#-local-setup--installation)
- [Docker Deployment](#-docker-deployment)
- [Deploying to Render](#-deploying-to-render)
- [Browser Compatibility](#-browser-compatibility)
- [Engineering Credits & License](#-engineering-credits--license)

---

## 🌿 Project Overview

**HarvestIQ** bridges agricultural plant pathology and on-field decision support. Built using Flask, PyTorch, Hugging Face Transformers, and LangChain, the platform allows users to upload foliage photographs of staple crops to receive immediate diagnostic evaluation.

The system combines computer vision predictions with local weather data (temperature and humidity fetched from Open-Meteo) and prompts an agronomic LLM (Gemini or Groq Llama 3.3) to generate step-by-step chemical and organic spray schedules, prevention protocols, and conversational support.

---

## ⚡ Key Features

| Feature | Details |
| :--- | :--- |
| 🔬 **Multi-Crop Pathology Scanner** | Deep classification across 4 staple crops: **Rice, Corn, Wheat, and Potato**, with fallback detection for non-supported or unconfident inputs (< 50% confidence). |
| 👁️ **ViT Attention Rollout Heatmaps** | Custom visual explainability engine rolling attention matrices across Vision Transformer blocks to draw contour-bounded lesion regions on the leaf. |
| 🌦️ **Microclimate Context Injection** | Ingests real-time ambient temperature and relative humidity into LLM prompts to prevent wash-off and recommend weather-appropriate systemic vs. contact treatments. |
| 🤖 **Dual-Provider LLM Core** | Dynamic engine supporting **Google Gemini (`gemini-1.5-flash`)** and **Groq (`llama-3.3-70b-versatile`)** via LangChain, complete with automated API quota fallback messaging. |
| 🗣️ **Multilingual & Voice Support** | Full translation pipeline and browser Web Speech API voice queries supporting **English**, **Hindi (हिंदी)**, and **Bengali (বাংলা)**. |
| 📄 **Prescription Sheet PDF Export** | Generates exportable diagnostic sheets client-side via `html2pdf.js` for on-field physical reference. |
| 🌌 **Dark Glassmorphic UI** | Responsive dark-mode interface styled with Tailwind CSS, custom status badges, confidence bars, and pulse indicators. |

---

## 🏗️ Core AI & ML Architecture

```text
               [ User / Farmer ]
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[ Leaf Photo (JPEG/PNG) ]    [ Voice / Text Chat Query ]
         │                           │
         │ (HTTP POST /analyze)      │ (HTTP POST /chat)
         ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       Flask Web Core                        │
│            Session State • Multipart File Ingestion         │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│ Vision Pipeline (HuggingFace)│ │  LangChain Reasoning Engine │
│ Model: wambugu71/...-vit     │ │  Providers: Gemini / Groq  │
│  - Eager Attention Rollout   │ │  - Multilingual Translation│
│  - Morphological OpenCV Mask │ │  - Weather Context Binding │
│  - Top-K Crop/Disease Parse  │ │  - Chat History Buffer     │
└──────────────┬───────────────┘ └────────────┬───────────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
           [ Real-Time UI Render & PDF Download ]
```

---

## 📁 Repository Structure

```text
DesesDetection/
├── app.py                            # Flask server, ViT rollout engine, and LangChain routes
├── Dockerfile                        # Containerized production environment definition
├── Hugging Face Training Script.py   # Model fine-tuning script for leaf pathology dataset
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies (Torch, Transformers, LangChain, etc.)
├── templates/
│   └── index.html                    # Glassmorphic UI, marked.js, Web Speech API & html2pdf
└── Deses/                            # Local Python virtual environment (ignored in git)
```

---

## 🔐 Environment Configuration

Create a `.env` file in the project root to configure runtime secrets and provider options:

```ini
# Flask Secret Key
SECRET_KEY=harvestiq_codoleox16_secret_key_2026

# Vision Transformer Model (Hugging Face ID or local path)
VISION_MODEL_ID=wambugu71/crop_leaf_diseases_vit

# LLM Provider Selection: "google" or "groq"
LLM_PROVIDER=google

# Google Gemini Configuration
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_MODEL=gemini-1.5-flash

# Groq Configuration (Optional fallback)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## 🔌 API Specification

### 1. Diagnostic Vision Engine (`/analyze`)

Receives an uploaded leaf photograph, runs ViT classification, generates an attention rollout bounding box overlay, and queries the LLM for tailored agronomic remedies.

- **URL:** `/analyze`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

#### Request Parameters
| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | `File (Binary)` | **Yes** | Image file (`.jpg`, `.jpeg`, `.png`) |
| `language` | `String` | No | Selected language (`English`, `Hindi`, `Bengali`) |
| `weather` | `String` | No | Ingested ambient weather (e.g. `"31°C, 78% Humidity"`) |

#### Response (`200 OK`)
```json
{
  "crop": "Potato",
  "disease": "Early Blight",
  "confidence": 0.942,
  "is_healthy": false,
  "attention_overlay": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "remedy": "### Visual Confirmation\nConcentric brown rings ('target board' spots) observed...\n\n### Actionable Treatment Plan\n* Spray **Mancozeb 75% WP** (2g/L).\n* Due to 78% humidity, repeat application after 7 days.\n\n### Preventative Protocols\n* Practice crop rotation and avoid overhead sprinkler irrigation.",
  "supported_crops": ["Corn", "Potato", "Rice", "Wheat"]
}
```

---

### 2. Conversational Agronomist (`/chat`)

Maintains session-based conversational history with HarvestBot to answer soil chemistry, crop nutrition, and pest questions.

- **URL:** `/chat`
- **Method:** `POST`
- **Content-Type:** `application/json`

#### Request Body
```json
{
  "message": "What should I spray on wheat showing brown leaf spots if rain is expected?",
  "language": "English",
  "weather": "26°C, 85% Humidity"
}
```

#### Response (`200 OK`)
```json
{
  "response": "With 85% humidity and rain expected, avoid contact sprays that wash off. Opt for a systemic fungicide like **Propiconazole 25% EC** (1 ml/L) mixed with a non-ionic wetting sticker, applied when the foliage is dry."
}
```

---

## 💻 Local Setup & Installation

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed
- Git installed
- A valid Google AI Studio (`GOOGLE_API_KEY`) or Groq (`GROQ_API_KEY`) API token

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/DesesDetection.git](https://github.com/your-username/DesesDetection.git)
cd DesesDetection
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv Deses
Deses\Scripts\activate

# Linux / macOS
python3 -m venv Deses
source Deses/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create your `.env` file as shown in the [Environment Configuration](#-environment-configuration) section.

### 5. Launch the Application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your web browser.

---

## 🐳 Docker Deployment

The repository includes a ready-to-run `Dockerfile` for containerized environments:

### 1. Build the Docker Image
```bash
docker build -t harvestiq:latest .
```

### 2. Run the Docker Container
```bash
docker run -d -p 5000:5000 \
  --env GOOGLE_API_KEY="your_api_key_here" \
  --name harvestiq-app \
  harvestiq:latest
```
Access the application at `http://localhost:5000`.

---

## ☁️ Deploying to Render

This application is configured for deployment on [Render](https://render.com/):

1. Push your repository to GitHub.
2. In the Render Dashboard, click **New +** → **Web Service**.
3. Connect your GitHub repository.
4. Set the configuration values:
   - **Environment:** `Python` or `Docker`
   - **Build Command (if using Python environment):**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command (if using Python environment):**
     ```bash
     gunicorn -w 1 -b 0.0.0.0:$PORT app:app
     ```
5. In **Environment Variables**, add:
   - `GOOGLE_API_KEY` = your active key
   - `LLM_PROVIDER` = `google`
   - `PYTHON_VERSION` = `3.10.12`
6. Click **Deploy Web Service**.

> **Note on Memory:** Running a PyTorch Vision Transformer pipeline in memory requires at least 1 GB to 2 GB of RAM on your hosting tier.

---

## 🧪 Browser Compatibility

| Feature | Google Chrome | Microsoft Edge | Mozilla Firefox | Apple Safari |
| :--- | :---: | :---: | :---: | :---: |
| **Glassmorphism & Layout** | ✅ Supported | ✅ Supported | ✅ Supported | ✅ Supported |
| **Geolocation Telemetry** | ✅ Supported | ✅ Supported | ✅ Supported | ✅ Supported |
| **PDF Prescription Generation** | ✅ Supported | ✅ Supported | ✅ Supported | ✅ Supported |
| **Speech-to-Text Input** | ✅ Native | ✅ Native | ⚠️ Configuration required | ⚠️ Partial |

---

## 👥 Engineering Credits & License

- **Lead Development:** Team CodoLeoX16
- **Vision Model Foundation:** `wambugu71/crop_leaf_diseases_vit` via Hugging Face Hub
- **Weather Telemetry:** Open-Meteo Geolocation API

This project is licensed under the [MIT License](LICENSE).
