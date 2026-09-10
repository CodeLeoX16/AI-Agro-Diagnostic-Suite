<div align="center">

# 🌾 HarvestIQ
### AI Agro Diagnostic Suite `v2.0 Universal`

An intelligent agronomic workstation combining deep neural vision, real-time microclimate sensing, and multilingual conversational AI to diagnose and treat staple crop diseases.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Render_Cloud-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://harvestiq-ai-powered-agriculture.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Flask-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tailwind CSS](https://img.shields.io/badge/UI-TailwindCSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br />

### 🌐 [Launch Live Web Application →](https://harvestiq-ai-powered-agriculture.onrender.com/)

</div>

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [API Contract Specification](#-api-contract-specification)
  - [1. Diagnostic Vision Engine (`/analyze`)](#1-diagnostic-vision-engine-analyze)
  - [2. Conversational Agronomist (`/chat`)](#2-conversational-agronomist-chat)
- [Local Setup & Installation](#-local-setup--installation)
- [Deployment on Render](#-deployment-on-render)
- [Browser Compatibility Matrix](#-browser-compatibility-matrix)
- [Team & Credits](#-team--credits)
- [License](#-license)

---

## 🌿 Overview

**HarvestIQ** bridges clinical plant pathology and on-field agronomic decision support. By analyzing foliar imagery against trained pathology classifiers and enriching the diagnosis with local microclimatic data (ambient temperature and relative humidity via Open-Meteo), HarvestIQ delivers actionable, dosage-accurate prescription sheets alongside an interactive voice-enabled AI copilot.

---

## ⚡ Key Features

| Feature | Description |
| :--- | :--- |
| 🔬 **Multi-Crop Pathology Scanner** | Real-time classification covering staple food crops: **Rice, Corn, Wheat, and Potato**. |
| 🎯 **Explainable Attention Mapping** | Renders dynamic activation overlays (`attention_overlay`) highlighting infected lesion boundaries. |
| 🌦️ **Microclimate Sensor Ingestion** | Automatically syncs device GPS with Open-Meteo APIs to contextualize disease risk against ambient temperature and humidity. |
| 🤖 **HarvestBot Agro-Copilot** | Localized LLM trained on soil chemistry, integrated pest management (IPM), and chemical/organic interventions. |
| 🗣️ **Multilingual & Voice-Enabled** | Native speech-to-text recognition and localization across **English**, **Hindi (हिंदी)**, and **Bengali (বাংলা)**. |
| 📄 **Prescription Sheet PDF Export** | Client-side DOM-to-PDF export pipeline powered by `html2pdf.js` for physical field use. |
| 🌌 **Dark Glassmorphic UI** | High-contrast, responsive interface crafted with custom utility palettes (`harvestDark`, `harvestGreen`, `harvestGold`). |

---

## 🏗️ System Architecture

```text
[ Farmer / Agronomist ]
        │
        ├── Uploads Leaf Image (JPG/PNG)
        ├── Captures Voice / Text Query
        └── Syncs Geolocation (Lat/Lon)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                 HarvestIQ Single-Page Application           │
│  Tailwind CSS • Web Speech API • Open-Meteo • Marked.js    │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
        POST /analyze                   POST /chat
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│      Vision Inference        │ │      Agronomic LLM        │
│  PyTorch / ONNX Classifier   │ │  Soil, Disease & Climate   │
│   Lesion / Attention Map     │ │    Prescription Prompts    │
└──────────────┬───────────────┘ └────────────┬───────────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
        [ Diagnostic Report & PDF Prescription ]
```

---

## 📁 Repository Structure

```text
HarvestIQ/
├── backend/
│   ├── app.py                 # Core application server (FastAPI / Flask)
│   ├── requirements.txt       # Python dependencies
│   ├── Procfile               # Deployment run process for Render
│   ├── services/
│   │   ├── classifier.py      # Neural vision inference & Grad-CAM pipeline
│   │   ├── assistant.py       # Conversational LLM wrapper & prompt logic
│   │   └── weather.py         # Microclimate fallback handlers
│   └── models/
│       ├── crop_weights.onnx  # Exported model weights
│       └── classes.json       # Crop and disease taxonomy mappings
├── frontend/
│   └── index.html             # Single-page client interface
├── assets/
│   └── preview.png            # UI preview graphics
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔌 API Contract Specification

### 1. Diagnostic Vision Engine (`/analyze`)

Analyzes uploaded crop foliage photos and returns disease identification, confidence scores, and remediation instructions.

- **Endpoint:** `/analyze`
- **HTTP Method:** `POST`
- **Encoding:** `multipart/form-data`

#### Request Payload
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | `Binary` | **Yes** | Foliar image file (`.png`, `.jpg`, `.jpeg`) |
| `language` | `String` | **Yes** | Selected locale: `English`, `Hindi`, or `Bengali` |
| `weather` | `String` | No | Real-time weather string (e.g., `"29°C, 82% Humidity"`) |

#### Success Response (`200 OK`)
```json
{
  "crop": "Potato",
  "disease": "Early Blight (Alternaria solani)",
  "confidence": 0.942,
  "is_healthy": false,
  "remedy": "### Diagnostic Overview\nLesions consistent with **Alternaria solani** observed.\n\n### Field Action Plan\n* Apply **Mancozeb 75% WP** (2.0 g/L water) or **Azoxystrobin 23% SC**.\n* Prune lower infected foliage to suppress spore splashback.\n* Eliminate furrow water stagnation immediately.",
  "attention_overlay": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "error": null
}
```

---

### 2. Conversational Agronomist (`/chat`)

Processes field questions and returns context-aware treatment, fertilizer, or soil guidelines.

- **Endpoint:** `/chat`
- **HTTP Method:** `POST`
- **Encoding:** `application/json`

#### Request Body
```json
{
  "message": "Which fungicide should I spray on potato early blight if rain is expected tonight?",
  "language": "English",
  "weather": "29°C, 82% Humidity"
}
```

#### Success Response (`200 OK`)
```json
{
  "response": "With high humidity and impending rainfall, prioritize a **systemic translaminar fungicide** (e.g., **Difenoconazole 25% EC** at 0.5 ml/L) over purely surface protectants like Mancozeb, as systemic options absorb rapidly and resist rain wash-off within 2 to 3 hours of dry application."
}
```

---

## 💻 Local Setup & Installation

### Prerequisites
- **Python 3.9+**
- **pip** and **git**

### 1. Clone Repository
```bash
git clone [https://github.com/your-username/HarvestIQ.git](https://github.com/your-username/HarvestIQ.git)
cd HarvestIQ
```

### 2. Configure Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Run Development Server
```bash
# If using FastAPI:
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# If using Flask:
python backend/app.py
```

Access the UI at `http://localhost:8000`.

---

## ☁️ Deployment on Render

This repository is pre-configured for deployment on [Render](https://render.com/):

1. Fork or push this repository to GitHub.
2. In the Render Dashboard, choose **New +** → **Web Service**.
3. Link your repository.
4. Configure service parameters:
   - **Environment:** `Python 3`
   - **Build Command:**
     ```bash
     pip install -r backend/requirements.txt
     ```
   - **Start Command:**
     ```bash
     gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.app:app
     ```
5. Click **Create Web Service** to launch your live instance.

---

## 🧪 Browser Compatibility Matrix

| Feature | Google Chrome | Mozilla Firefox | Apple Safari | Microsoft Edge |
| :--- | :---: | :---: | :---: | :---: |
| **Glassmorphic Theme & Layout** | Full | Full | Full | Full |
| **HTML5 Geolocation Ingestion** | Full | Full | Full | Full |
| **One-Click PDF Prescription** | Full | Full | Full | Full |
| **Web Speech Voice Input** | Full | Partial / Flag | Partial | Full |

---

## 👥 Team & Credits

Engineered with precision by **Team CodoLeoX16**.

- Open-source contributions, bug reports, and dataset enrichments are welcome.
- Please open an **Issue** or submit a **Pull Request** following the contribution guidelines.

---

## 📄 License

This project is open-source software licensed under the **[MIT License](LICENSE)**.
