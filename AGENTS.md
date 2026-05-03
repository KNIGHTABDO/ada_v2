# AGENTS.md

## Project Overview
**ADA V2 (Advanced Design Assistant)** is a multimodal desktop assistant designed for high-performance interaction. It combines real-time voice, vision, and desktop automation to help the user (ABDO) with design, CAD, and daily tasks.

## Core Tech Stack
- **Brain**: Groq API (Llama 3.3-70b-versatile).
- **Vision**: Moondream Cloud API (v1 Query).
- **Voice**: Deepgram (Aura-Asteria for TTS, Nova-2 for STT).
- **Automation**: PyAutoGUI for hardware-level keyboard/mouse simulation.
- **Backend**: Python (FastAPI + Socket.io).
- **Frontend**: React + Vite + Electron.

## ⚠️ Critical Agent Rules (DON'T BREAK THESE)

### 1. Groq Tool Schemas (The "400 Error" Trap)
Groq is extremely strict about JSON schemas. 
- **Rule**: ALL types in `parameters` MUST be **lowercase** (`string`, `object`, `integer`, `boolean`). 
- **Failure**: Using `"STRING"` or `"OBJECT"` will cause a `400 BadRequestError` and crash the assistant.

### 2. Async Non-Blocking Code
The Python backend runs on an `asyncio` event loop.
- **Rule**: Never use `time.sleep()`. Always use `await asyncio.sleep()`.
- **Failure**: Blocking the loop for even 1 second will cause the Socket.io connection to time out, leading to `ERR_CONNECTION_REFUSED` in the browser.

### 3. Spotify Automation
The Spotify "play" logic is a precision macro. 
- **Method**: It uses `os.startfile` with a search URI, waits 7 seconds via `asyncio.sleep`, then uses `pyautogui.press(['down', 'enter', 'enter'])`.
- **Note**: Ensure `pyautogui` has access to the desktop session.

### 4. Vision (Moondream)
- **Data Flow**: The frontend sends base64 image frames to the backend in `self._latest_image_payload`.
- **Usage**: The `see` tool grabs the latest frame and sends it to `https://api.moondream.ai/v1/query`.

## Setup & Dev Commands
- **Install Deps**: `npm install` (Frontend) and use the `venv_ada` environment for Python.
- **Run All**: `npm run dev` (Starts Vite, Electron, and the Python Backend concurrently).
- **Backend Port**: `8000` (FastAPI/Socket.io).
- **Frontend Port**: `5173` (Vite).

## File Map
- `backend/ada.py`: The heart of the assistant (Brain, Tool execution, Audio Loop).
- `backend/server.py`: FastAPI server and Socket.io routing.
- `backend/tools.py`: Secondary tool definitions.
- `src/App.jsx`: Frontend logic, Webcam handling, and UI.
