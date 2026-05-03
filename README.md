# 🤖 ADA V2: The Advanced Design Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Brain-Groq%20%7C%20Copilot-orange?style=for-the-badge&logo=openai" alt="Brain">
  <img src="https://img.shields.io/badge/Voice-Deepgram%20%2B%20ElevenLabs-FF69B4?style=for-the-badge&logo=elevenlabs" alt="Voice">
  <img src="https://img.shields.io/badge/Vision-Moondream-blue?style=for-the-badge&logo=google-cloud" alt="Vision">
</p>

**ADA V2** is a multimodal, desktop-native AI assistant designed for high-performance interaction. Unlike standard chatbots, ADA combines real-time voice, computer vision, and hardware-level automation to serve as a true "Jarvis-like" companion for design, engineering, and daily productivity.

---

## 🚀 The Core Philosophy

ADA V2 is built on the principle of **High-Velocity Intelligence**. By separating "Utility" (speed) from "HD Expression" (fidelity), ADA delivers a response experience that feels alive, reactive, and emotionally intelligent.

### 🎙️ Dual-Engine Voice Architecture
ADA intelligently routes audio synthesis between two industry-leading engines:
- **⚡ Deepgram Aura-2 (Fast Mode)**: For instant utility tasks, command confirmations, and status updates. Sub-100ms latency.
- **🎭 ElevenLabs v3 (HD Mode)**: For complex explanations, storytelling, and emotional interactions. Responds to inline audio tags like `[laughs]`, `[whispers]`, and `[excited]` to provide cinematic-quality prosody.

---

## 🌟 Key Capabilities

### 🧊 Autonomous CAD Generation
Generate and iterate on 3D designs using only your voice. 
- Powered by `build123d` for high-precision parametric modeling.
- Direct export to STL.
- Multi-turn iteration: *"Make the bolt head thicker,"* or *"Add a 5mm hole to the center."*

### 👁️ Multimodal Vision (Moondream)
ADA doesn't just talk; she sees.
- **Desktop Vision**: Capture and analyze your screen to help you debug code, explain UI elements, or find specific files.
- **Camera Vision**: Describe the real world, recognize objects, or read text from a physical paper.

### 🖐️ Gesture-Driven "Minority Report" UI
Interact with your desktop using hand movements.
- **Pinch to Click**: Confirm actions without touching your mouse.
- **Fist to Drag**: Grab and move application windows in real-time.
- **Open Palm to Release**: Drop windows exactly where you want them.

### 🌐 Autonomous Web Agent
Need to find something? ADA can browse the web for you.
- Uses **Playwright** to autonomously navigate websites, scroll, click, and extract information.
- Perfect for research, shopping price comparisons, or technical documentation lookups.

### 🔒 Face Authentication & Security
- **Biometric Login**: Secure the entire assistant with MediaPipe-powered face recognition.
- **Safety Confirmations**: Critical actions (like writing files or launching browsers) require your explicit verbal or UI confirmation.

---

## 🏗️ Technical Stack

- **Frontend**: React 18, Vite, Tailwind CSS, Electron.
- **Backend**: Python 3.11 (FastAPI + Socket.io).
- **AI Brain**: Groq (Llama 3.3-70b) & GitHub Copilot API.
- **Vision**: Moondream Cloud API.
- **Audio**: PyAudio (Streaming), Deepgram Aura-2, ElevenLabs v3.
- **Automation**: PyAutoGUI & Playwright.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.11** (Highly recommended to use Conda/Miniconda).
- **Node.js 18+**.
- **Windows 10/11** (Optimized for Windows Terminal).

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ada_v2.git
cd ada_v2

# Create and activate Python environment
conda create -n venv_ada python=3.11 -y
conda activate venv_ada

# Install dependencies
pip install -r requirements.txt
playwright install chromium
npm install
```

### 3. Configuration (`.env`)
Create a `.env` file in the root directory:
```env
# Intelligence
GROQ_API_KEY=your_groq_key
GITHUB_COPILOT_TOKEN=your_token_here (Optional)

# Voice
DEEPGRAM_API_KEY=your_deepgram_key
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id_here

# Vision
MOONDREAM_API_KEY=your_moondream_key
```

### 4. Running the Assistant
```bash
# Start everything concurrently
npm run dev
```

---

## 📂 Project Structure

```
ada_v2/
├── backend/            # The Brain (Python logic)
│   ├── ada.py          # Orchestration & TTS Routing
│   ├── server.py       # Socket.io & API Server
│   ├── cad_agent.py    # build123d Logic
│   └── web_agent.py    # Playwright Automation
├── src/                # The Body (React Frontend)
│   ├── components/     # UI Windows & Visualizers
│   └── App.jsx         # Main Interface logic
├── electron/           # Desktop Shell
└── projects/           # Persistent User Data
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  <strong>Built for ABDO by Antigravity</strong><br>
  <em>High-Performance Multimodal Autonomy</em>
</p>
