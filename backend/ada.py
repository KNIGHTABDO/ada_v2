import asyncio
import base64
import io
import os
import sys
import traceback
import io

# Force UTF-8 for Windows Terminal to avoid Arabic/Unicode print crashes
if sys.platform == 'win32' and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from dotenv import load_dotenv
import cv2
import pyaudio
import PIL.Image
import mss
import argparse
import math
import struct
import time

import httpx
import uuid
import pygame
from groq import AsyncGroq
from google import genai
from google.genai import types
import wave

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

from tools import tools_list

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL_BRAIN = "gpt-4o" # Default to Copilot
DEFAULT_BRAIN_FALLBACKS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "allam-2-7b",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
    "groq/compound-mini"
]
MODEL_VISION = "moondream"
MODEL_VOICE = "deepgram-aura-2" # Primary utility voice
MODEL_VOICE_HD = "eleven-v3"   # High-definition expressive voice
DEFAULT_MODE = "camera"
USE_PIPELINE = True

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
moondream_api_key = os.getenv("MOONDREAM_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
github_copilot_token = os.getenv("GITHUB_COPILOT_TOKEN")

# Global cache for Copilot token
_copilot_runtime_token = None
_copilot_token_expires_at = 0

def _parse_model_fallbacks():
    raw = os.getenv("GROQ_MODEL_FALLBACKS", "").strip()
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]

if not groq_api_key:
    print("[ADA ERROR] GROQ_API_KEY not found in .env!")
if not deepgram_api_key:
    print("[ADA ERROR] DEEPGRAM_API_KEY not found in .env!")
if not moondream_api_key:
    print("[ADA ERROR] MOONDREAM_API_KEY not found in .env!")
if not google_api_key:
    print("[ADA ERROR] GOOGLE_API_KEY not found in .env!")

client = AsyncGroq(api_key=groq_api_key)
google_client = genai.Client(api_key=google_api_key) if google_api_key else None

# Function definitions
generate_cad = {
    "name": "generate_cad",
    "description": "Generates a 3D CAD model based on a prompt.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The description of the object to generate."}
        },
        "required": ["prompt"]
    },
    "behavior": "NON_BLOCKING"
}

run_web_agent = {
    "name": "run_web_agent",
    "description": "Opens a web browser and performs a task according to the prompt.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The detailed instructions for the web browser agent."}
        },
        "required": ["prompt"]
    },
    "behavior": "NON_BLOCKING"
}

create_project_tool = {
    "name": "create_project",
    "description": "Creates a new project folder to organize files.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name of the new project."}
        },
        "required": ["name"]
    }
}

switch_project_tool = {
    "name": "switch_project",
    "description": "Switches the current active project context.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name of the project to switch to."}
        },
        "required": ["name"]
    }
}

list_projects_tool = {
    "name": "list_projects",
    "description": "Lists all available projects.",
    "parameters": {
        "type": "object",
        "properties": {},
    }
}

list_smart_devices_tool = {
    "name": "list_smart_devices",
    "description": "Lists all available smart home devices (lights, plugs, etc.) on the network.",
    "parameters": {
        "type": "object",
        "properties": {},
    }
}

control_light_tool = {
    "name": "control_light",
    "description": "Controls a smart light device.",
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "The IP address of the device to control. Always prefer the IP address over the alias for reliability."
            },
            "action": {
                "type": "string",
                "description": "The action to perform: 'turn_on', 'turn_off', or 'set'."
            },
            "brightness": {
                "type": "integer",
                "description": "Optional brightness level (0-100)."
            },
            "color": {
                "type": "string",
                "description": "Optional color name (e.g., 'red', 'cool white') or 'warm'."
            }
        },
        "required": ["target", "action"]
    }
}

discover_printers_tool = {
    "name": "discover_printers",
    "description": "Discovers 3D printers available on the local network.",
    "parameters": {
        "type": "object",
        "properties": {},
    }
}

print_stl_tool = {
    "name": "print_stl",
    "description": "Prints an STL file to a 3D printer. Handles slicing the STL to G-code and uploading to the printer.",
    "parameters": {
        "type": "object",
        "properties": {
            "stl_path": {"type": "string", "description": "Path to STL file, or 'current' for the most recent CAD model."},
            "printer": {"type": "string", "description": "Printer name or IP address."},
            "profile": {"type": "string", "description": "Optional slicer profile name."}
        },
        "required": ["stl_path", "printer"]
    }
}

get_print_status_tool = {
    "name": "get_print_status",
    "description": "Gets the current status of a 3D printer including progress, time remaining, and temperatures.",
    "parameters": {
        "type": "object",
        "properties": {
            "printer": {"type": "string", "description": "Printer name or IP address."}
        },
        "required": ["printer"]
    }
}

iterate_cad_tool = {
    "name": "iterate_cad",
    "description": "Modifies or iterates on the current CAD design based on user feedback. Use this when the user asks to adjust, change, modify, or iterate on the existing 3D model (e.g., 'make it taller', 'add a handle', 'reduce the thickness').",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The changes or modifications to apply to the current design."}
        },
        "required": ["prompt"]
    },
    "behavior": "NON_BLOCKING"
}

terminate_application_tool = {
    "name": "terminate_application",
    "description": "Closes an application using the OS process manager (no UI focus required).",
    "parameters": {
        "type": "object",
        "properties": {
            "app_name": {"type": "string", "description": "App name or process name (e.g. spotify, Spotify.exe)."},
            "force": {"type": "boolean", "description": "Force close the app (default true)."}
        },
        "required": ["app_name"]
    }
}

open_application_tool = {
    "name": "open_application",
    "description": "Opens an application on the user's computer.",
    "parameters": {
        "type": "object",
        "properties": {
            "app_name": {"type": "string", "description": "The name of the app to open (e.g. spotify, chrome, notepad, calculator)."},
            "wait": {"type": "integer", "description": "Optional seconds to wait after opening (default 5)."},
            "url": {"type": "string", "description": "Optional URL to open (e.g. https://example.com)."},
            "query": {"type": "string", "description": "Optional search query for browsers (e.g. 'Lionel Messi')."}
        },
        "required": ["app_name"]
    }
}

see_tool = {
    "name": "see",
    "description": "Look at the camera and describe what is visible.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The question or description to look for."}
        },
        "required": ["prompt"]
    }
}

play_music_tool = {
    "name": "play_music",
    "description": "Searches for and plays a specific song or artist on Spotify.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The song name, artist, or album to play (e.g. 'Billie Jean by Michael Jackson')."}
        },
        "required": ["query"]
    }
}

get_ui_coordinates_tool = {
    "name": "get_ui_coordinates",
    "description": "Finds the exact coordinates of a UI element on the screen (e.g., 'the search bar', 'the play button'). Returns normalized coordinates (0-1).",
    "parameters": {
        "type": "object",
        "properties": {
            "object_name": {"type": "string", "description": "The name of the UI element to find."}
        },
        "required": ["object_name"]
    }
}

see_desktop_tool = {
    "name": "see_desktop",
    "description": "Captures a screenshot of the desktop and describes it. Use this to 'see' what is happening on the screen, find buttons, or verify if an app is open.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "What to look for on the screen (e.g., 'where is the search bar?', 'is Spotify playing?')."}
        },
        "required": ["prompt"]
    }
}

mouse_action = {
    "name": "mouse_action",
    "description": "Perform mouse actions like clicking or moving. Coordinates are in percentages (0-100).",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "The action: 'click', 'double_click', 'right_click', or 'move'."},
            "x": {"type": "integer", "description": "X coordinate in percentage (0-100)."},
            "y": {"type": "integer", "description": "Y coordinate in percentage (0-100)."}
        },
        "required": ["action", "x", "y"]
    }
}

keyboard_action = {
    "name": "keyboard_action",
    "description": "Perform keyboard actions like typing or pressing special keys.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "The action: 'type' or 'press'."},
            "text": {"type": "string", "description": "The text to type or the key to press (e.g., 'enter', 'tab', 'ctrl+c')."}
        },
        "required": ["action", "text"]
    }
}

# Core Speed Demon Tools (Autonomous UI Agent Edition)
raw_tools = [
    see_tool,
    see_desktop_tool,
    get_ui_coordinates_tool,
    mouse_action,
    keyboard_action,
    play_music_tool,
    open_application_tool,
    terminate_application_tool,
    generate_cad,
    iterate_cad_tool,
]

FOCUS_TOOL_NAMES = {
    "open_application",
    "play_music",
    "see_desktop",
    "get_ui_coordinates",
    "mouse_action",
    "keyboard_action",
}

import copy

def _recursive_lowercase_type(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "type" and isinstance(v, str):
                new_obj[k] = v.lower()
            else:
                new_obj[k] = _recursive_lowercase_type(v)
        return new_obj
    elif isinstance(obj, list):
        return [_recursive_lowercase_type(x) for x in obj]
    else:
        return obj

groq_tools = []
for t_orig in raw_tools:
    t = copy.deepcopy(t_orig)
    if "behavior" in t:
        del t["behavior"]
    groq_tools.append({"type": "function", "function": t})

SYSTEM_PROMPT = (
    "You are ADA, an autonomous computer assistant for ABDO (Sir). "
    "CRITICAL: Never assume the results of a search or a tool action based on your internal knowledge. "
    "You MUST call 'see_desktop' to verify what is on the screen after searching or opening a URL before providing information. "
    "Your verbal responses must be optimized for TTS: Avoid emojis, asterisks (*), hashtags (#), or markdown formatting. "
    "VOICE ROUTING (MANDATORY): You MUST decide which voice to use for each response. "
    "Start EVERY response with one of these EXACT tags: "
    "- [VOICE: FAST]: Use ONLY for short actions, status updates, or one-sentence confirmations. "
    "- [VOICE: HD]: Use for ALL stories, explanations, long chats, or when using expressive audio tags. "
    "Example 1: '[VOICE: FAST] [happy] Switching to project Alpha, Sir.' "
    "Example 2: '[VOICE: HD] [excited] I found the design! [whispers] It looks perfect. [laughs]'"
    "VOICE PERSONALITY: You have an expressive Jarvis-like voice. Use these audio tags in [VOICE: HD] mode: "
    "- [excited], [whispers], [happy], [laughs], [slow], [fast]. "
    "Be proactive, witty, and address ABDO as Sir."
)

# Tool schemas for prompt-based tool calling (sent as text, not native Groq tools)
_PIPELINE_TOOL_SCHEMAS = """
You have access to the following tools. To use a tool, output ONLY a JSON object on a single line in this exact format:
{"tool": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}

Available tools:
- open_application: Opens an application. Args: {"app_name": "spotify|chrome|notepad|calculator|...", "wait": 5, "url": "https://...", "query": "search terms"} — wait is seconds to let the app load (default 5, use 10+ for slow PCs)
- terminate_application: Closes an application using the OS process manager. Args: {"app_name": "spotify|Spotify.exe", "force": true}
- play_music: Searches and plays a song on Spotify. Args: {"query": "song name or artist"}
- see: Looks at the camera and describes what is visible. Args: {"prompt": "what to look for"}
- see_desktop: Captures a screenshot and describes it. Args: {"prompt": "what to look for on screen"}
- get_ui_coordinates: Finds coordinates of a UI element. Args: {"object_name": "the element name"}
- mouse_action: Performs mouse actions. Args: {"action": "click|double_click|right_click|move", "x": 0-100, "y": 0-100}
- keyboard_action: Performs keyboard actions. Args: {"action": "type|press", "text": "text or key name"}
- create_project: Creates a new project folder. Args: {"name": "project name"}

IMPORTANT RULES:
- If you need to use a tool, output ONLY the JSON tool call. Do not add any other text.
- If you don't need a tool, respond normally with text.
- Do NOT use XML tags like <function=...>. Always use the JSON format above.
- After you see the tool result, respond with a natural language message to the user.
- For any desktop action (open apps, clicks, typing, playback), you MUST call a tool. Do not claim completion without tool results.
- For Spotify playback, call play_music and only confirm after tool results.
- For browser searches, call open_application with app_name "chrome" and query. Do not just open the app.
- For closing apps, prefer terminate_application over UI clicks.
"""

pya = pyaudio.PyAudio()

from cad_agent import CadAgent
from web_agent import WebAgent
from kasa_agent import KasaAgent
from printer_agent import PrinterAgent

async def _get_copilot_token():
    global _copilot_runtime_token, _copilot_token_expires_at
    now = time.time()
    
    if _copilot_runtime_token and _copilot_token_expires_at > now + 300: # 5 min buffer
        return _copilot_runtime_token

    if not github_copilot_token:
        raise Exception("GITHUB_COPILOT_TOKEN is missing")

    for attempt in range(3):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.github.com/copilot_internal/v2/token",
                    headers={
                        "Authorization": f"Bearer {github_copilot_token}",
                        "User-Agent": "GithubCopilot/1.155.0",
                        "Accept": "application/json"
                    },
                    timeout=30.0
                )
                resp.raise_for_status()
                data = resp.json()
                _copilot_runtime_token = data.get("token")
                
                # Handle expires_at safely
                try:
                    exp = data.get("expires_at", 0)
                    _copilot_token_expires_at = int(exp) if exp else int(time.time() + 1800)
                except (ValueError, TypeError):
                    _copilot_token_expires_at = int(time.time() + 1800)
                
                print(f"[ADA DEBUG] [AUTH] Refreshed Copilot token. Token length: {len(_copilot_runtime_token) if _copilot_runtime_token else 0}")
                return _copilot_runtime_token

        except Exception as e:
            if attempt == 2:
                print(f"[ADA ERROR] Failed to fetch Copilot token after 3 attempts: {e}")
                raise e
            print(f"[ADA DEBUG] [WARN] Copilot token attempt {attempt+1} failed: {e}. Retrying...")
            await asyncio.sleep(2)

async def _get_copilot_models():
    """Fetch exact available models from GitHub Copilot API."""
    try:
        token = await _get_copilot_token()
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.githubcopilot.com/models",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Editor-Version": "vscode/1.85.0",
                    "Editor-Plugin-Version": "copilot/1.155.0",
                    "User-Agent": "GithubCopilot/1.155.0",
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
            print(f"[ADA DEBUG] Copilot models response: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data.get("data"), list):
                    # Sort to put common models first
                    raw_models = data["data"]
                    print(f"[ADA DEBUG] Found {len(raw_models)} raw models")
                    formatted = []
                    for m in raw_models:
                        m_id = m.get("id")
                        # Add provider info for clarity
                        provider = m.get("capabilities", {}).get("family", "model")
                        name = m.get("name", m_id)
                        formatted.append({
                            "id": m_id,
                            "name": f"{name} ({provider})" if provider else name
                        })
                    return formatted
                else:
                    print(f"[ADA DEBUG] Copilot response data invalid: {data}")
            else:
                print(f"[ADA DEBUG] Copilot API error response: {resp.text}")
    except Exception as e:
        print(f"[ADA ERROR] Failed to fetch Copilot models: {e}")
        traceback.print_exc()
    
    # Fallback if API fails
    return [
        {"id": "gpt-4o", "name": "GPT-4o (openai)"},
        {"id": "claude-3.5-sonnet", "name": "Claude 3.5 Sonnet (anthropic)"},
        {"id": "gpt-4", "name": "GPT-4 (openai)"},
        {"id": "o1-preview", "name": "o1-preview (openai)"},
        {"id": "o1-mini", "name": "o1-mini (openai)"}
    ]

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, on_audio_data=None, on_video_frame=None, on_cad_data=None, on_web_data=None, on_transcription=None, on_tool_confirmation=None, on_cad_status=None, on_cad_thought=None, on_project_update=None, on_device_update=None, on_error=None, on_tts_state=None, on_focus_mode=None, input_device_index=None, input_device_name=None, output_device_index=None, kasa_agent=None):
        self.video_mode = video_mode
        self.on_audio_data = on_audio_data
        self.on_video_frame = on_video_frame
        self.on_cad_data = on_cad_data
        self.on_web_data = on_web_data
        self.on_transcription = on_transcription
        self.on_tool_confirmation = on_tool_confirmation 
        self.on_cad_status = on_cad_status
        self.on_cad_thought = on_cad_thought
        self.on_project_update = on_project_update
        self.on_device_update = on_device_update
        self.on_error = on_error
        self.on_tts_state = on_tts_state
        self.on_focus_mode = on_focus_mode
        self.input_device_index = input_device_index
        self.input_device_name = input_device_name
        self.output_device_index = output_device_index

        self.audio_in_queue = None
        self.out_queue = None
        self.paused = False

        self.chat_buffer = {"sender": None, "text": ""} # For aggregating chunks
        
        # Track last transcription text to calculate deltas (Gemini sends cumulative text)
        self._last_input_transcription = ""
        self._last_output_transcription = ""

        self.audio_in_queue = None
        self.out_queue = None
        self.paused = False

        self.session = None
        
        # Create CadAgent with thought callback
        def handle_cad_thought(thought_text):
            if self.on_cad_thought:
                self.on_cad_thought(thought_text)
        
        def handle_cad_status(status_info):
            if self.on_cad_status:
                self.on_cad_status(status_info)
        
        self.cad_agent = CadAgent(on_thought=handle_cad_thought, on_status=handle_cad_status)
        self.web_agent = WebAgent()
        self.kasa_agent = kasa_agent if kasa_agent else KasaAgent()
        self.printer_agent = PrinterAgent()
        
        # Pipeline State
        self.audio_buffer = [] # To accumulate audio for Lite Brain
        self.is_processing = False # To avoid overlapping turns
        self.chat_history = [] # For Brain context in pipeline mode
        self.client = client # Ensure pipeline has access to the global client

        self.send_text_task = None
        self.stop_event = asyncio.Event()
        
        self.stop_event = asyncio.Event()
        
        self.permissions = {} # Default Empty (Will treat unset as True)
        self._pending_confirmations = {}
        self._focus_mode_count = 0
        self._model_fallbacks = _parse_model_fallbacks() or list(DEFAULT_BRAIN_FALLBACKS)
        self._current_brain_model = MODEL_BRAIN
        self.stt_available = True
        self._stt_error_reported = False
        try:
            import speech_recognition as _sr
        except Exception as e:
            self.stt_available = False
            print(f"[ADA ERROR] SpeechRecognition not available: {e}")

        # Video buffering state
        self._latest_image_payload = None
        # VAD State
        self._is_speaking = False
        self._silence_start_time = None
        
        # Initialize ProjectManager
        from project_manager import ProjectManager
        # Assuming we are running from backend/ or root? 
        # Using abspath of current file to find root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # If ada.py is in backend/, project root is one up
        project_root = os.path.dirname(current_dir)
        self.project_manager = ProjectManager(project_root)
        
        # Sync Initial Project State
        if self.on_project_update:
            # We need to defer this slightly or just call it. 
            # Since this is init, loop might not be running, but on_project_update in server.py uses asyncio.create_task which needs a loop.
            # We will handle this by calling it in run() or just print for now.
            pass

    def flush_chat(self):
        """Forces the current chat buffer to be written to log."""
        if self.chat_buffer["sender"] and self.chat_buffer["text"].strip():
            self.project_manager.log_chat(self.chat_buffer["sender"], self.chat_buffer["text"])
            self.chat_buffer = {"sender": None, "text": ""}
        # Reset transcription tracking for new turn
        self._last_input_transcription = ""
        self._last_output_transcription = ""

    def update_permissions(self, new_perms):
        print(f"[ADA DEBUG] [CONFIG] Updating tool permissions: {new_perms}")
        self.permissions.update(new_perms)

    def set_paused(self, paused):
        self.paused = paused

    def stop(self):
        self.stop_event.set()
        
    def resolve_tool_confirmation(self, request_id, confirmed):
        print(f"[ADA DEBUG] [RESOLVE] resolve_tool_confirmation called. ID: {request_id}, Confirmed: {confirmed}")
        if request_id in self._pending_confirmations:
            future = self._pending_confirmations[request_id]
            if not future.done():
                print(f"[ADA DEBUG] [RESOLVE] Future found and pending. Setting result to: {confirmed}")
                future.set_result(confirmed)
            else:
                 print(f"[ADA DEBUG] [WARN] Request {request_id} future already done. Result: {future.result()}")
        else:
            print(f"[ADA DEBUG] [WARN] Confirmation Request {request_id} not found in pending dict. Keys: {list(self._pending_confirmations.keys())}")

    def clear_audio_queue(self):
        """Clears the queue of pending audio chunks to stop playback immediately."""
        try:
            count = 0
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()
                count += 1
            if count > 0:
                print(f"[ADA DEBUG] [AUDIO] Cleared {count} chunks from playback queue due to interruption.")
        except Exception as e:
            print(f"[ADA DEBUG] [ERR] Failed to clear audio queue: {e}")

    async def _set_focus_mode(self, active):
        if not self.on_focus_mode:
            return
        if active:
            self._focus_mode_count += 1
            if self._focus_mode_count == 1:
                self.on_focus_mode(True)
                await asyncio.sleep(0.35)
        else:
            if self._focus_mode_count > 0:
                self._focus_mode_count -= 1
                if self._focus_mode_count == 0:
                    self.on_focus_mode(False)
                    await asyncio.sleep(0.2)

    def _find_window_by_title(self, keywords):
        if sys.platform != "win32":
            return None
        if isinstance(keywords, str):
            keywords = [keywords]
        keywords = [k.lower() for k in keywords if k]
        if not keywords:
            return None

        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            EnumWindows = user32.EnumWindows
            EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            GetWindowTextW = user32.GetWindowTextW
            GetWindowTextLengthW = user32.GetWindowTextLengthW
            IsWindowVisible = user32.IsWindowVisible

            matches = []

            def _enum_handler(hwnd, lparam):
                if not IsWindowVisible(hwnd):
                    return True
                length = GetWindowTextLengthW(hwnd)
                if length == 0:
                    return True
                buffer = ctypes.create_unicode_buffer(length + 1)
                GetWindowTextW(hwnd, buffer, length + 1)
                title = buffer.value
                if title and any(k in title.lower() for k in keywords):
                    matches.append(hwnd)
                return True

            EnumWindows(EnumWindowsProc(_enum_handler), 0)
            return matches[0] if matches else None
        except Exception:
            return None

    async def _focus_window_by_title(self, keywords, timeout=6.0, interval=0.2):
        if sys.platform != "win32":
            return False
        import time
        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            hwnd = self._find_window_by_title(keywords)
            if hwnd:
                try:
                    import ctypes
                    user32 = ctypes.windll.user32
                    SW_RESTORE = 9
                    user32.ShowWindow(hwnd, SW_RESTORE)
                    user32.SetForegroundWindow(hwnd)
                    await asyncio.sleep(0.2)
                    return True
                except Exception:
                    return False
            await asyncio.sleep(interval)
        return False

    def _find_chrome_executable(self):
        if sys.platform != "win32":
            return None
        candidates = [
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    async def _point_ui_element(self, object_name):
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                img = PIL.Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{img_data}"
            url = "https://api.moondream.ai/v1/point"
            headers = {"X-Moondream-Auth": moondream_api_key}
            payload = {"image_url": data_uri, "object": f"the exact center of the {object_name}"}
            async with httpx.AsyncClient() as v_client:
                v_resp = await v_client.post(url, headers=headers, json=payload, timeout=30.0)
            if v_resp.status_code == 200:
                points = v_resp.json().get("points", [])
                if points:
                    pt = points[0]
                    return (pt.get("x"), pt.get("y"))
        except Exception:
            pass
        return None

    async def _click_ui_element(self, object_name, retries=2):
        import pyautogui
        for _ in range(max(1, retries)):
            point = await self._point_ui_element(object_name)
            if point and point[0] is not None and point[1] is not None:
                screen_w, screen_h = pyautogui.size()
                target_x = int(point[0] * screen_w)
                target_y = int(point[1] * screen_h)
                pyautogui.moveTo(target_x, target_y, duration=0.2)
                pyautogui.click(target_x, target_y)
                await asyncio.sleep(0.2)
                return True
            await asyncio.sleep(0.4)
        return False

    def _is_rate_limit_error(self, error):
        name = type(error).__name__.lower()
        msg = str(error).lower()
        return (
            "ratelimit" in name
            or "rate limit" in msg
            or "rate_limit" in msg
            or "429" in msg
        )

    async def _create_chat_completion(self, messages, temperature=0.3, max_tokens=1024, tools=None):
        candidates = [self._current_brain_model] + [
            m for m in self._model_fallbacks if m != self._current_brain_model
        ]
        last_error = None

        for model in candidates:
            # Route to Groq only for specific known Groq IDs, otherwise default to Copilot
            is_groq = any(q in model.lower() for q in ["llama", "mixtral", "gemma", "whisper", "deepseek"])
            is_copilot = not is_groq
            
            print(f"[ADA DEBUG] Routing model: {model} | is_copilot: {is_copilot} | is_groq: {is_groq}")
            
            try:
                if is_copilot:
                    token = await _get_copilot_token()
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Editor-Version": "vscode/1.85.0",
                        "Editor-Plugin-Version": "copilot/1.155.0",
                        "User-Agent": "GithubCopilot/1.155.0",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    if tools:
                        payload["tools"] = tools

                    async with httpx.AsyncClient() as c_client:
                        resp = await c_client.post(
                            "https://api.githubcopilot.com/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=30.0
                        )
                        if resp.status_code == 401 or resp.status_code == 429:
                            raise Exception(f"Copilot API error: {resp.status_code} - {resp.text}")
                        resp.raise_for_status()
                        
                        data = resp.json()
                        # Construct a mock object matching Groq's return type for seamless integration
                        class MockFunction:
                            def __init__(self, name, arguments):
                                self.name = name
                                self.arguments = arguments
                        class MockToolCall:
                            def __init__(self, id, function):
                                self.id = id
                                self.function = function
                        class MockMessage:
                            def __init__(self, content, tool_calls=None):
                                self.content = content
                                self.tool_calls = tool_calls
                        class MockChoice:
                            def __init__(self, message):
                                self.message = message
                        class MockResponse:
                            def __init__(self, choices):
                                self.choices = choices
                        
                        m_msg = data["choices"][0]["message"]
                        t_calls = None
                        if "tool_calls" in m_msg:
                            t_calls = [
                                MockToolCall(tc["id"], MockFunction(tc["function"]["name"], tc["function"]["arguments"]))
                                for tc in m_msg["tool_calls"]
                            ]
                        
                        result = MockResponse([MockChoice(MockMessage(m_msg.get("content"), tool_calls=t_calls))])
                else:
                    # Fallback to Groq natively
                    result = await self.client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools
                    )
                    
                if model != self._current_brain_model:
                    print(f"[ADA DEBUG] [BRAIN] Switched model to {model}")
                    self._current_brain_model = model
                return result
            except Exception as e:
                last_error = e
                print(f"[ADA DEBUG] [BRAIN] Error with {model}: {e}")
                if self._is_rate_limit_error(e) or "401" in str(e):
                    print(f"[ADA DEBUG] [BRAIN] Model {model} failed. Trying fallback...")
                    if self.on_error:
                        self.on_error(f"Error on {model}. Falling back to another model.")
                    continue
                raise

        raise last_error

    async def send_frame(self, frame_data):
        # Update the latest frame payload
        if isinstance(frame_data, bytes):
            b64_data = base64.b64encode(frame_data).decode('utf-8')
        else:
            b64_data = frame_data 

        # Store as the designated "next frame to send"
        self._latest_image_payload = {"mime_type": "image/jpeg", "data": b64_data}
        # No event signal needed - listen_audio pulls it

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg, end_of_turn=False)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()

        # Resolve Input Device by Name if provided
        resolved_input_device_index = None
        
        if self.input_device_name:
            print(f"[ADA] Attempting to find input device matching: '{self.input_device_name}'")
            count = pya.get_device_count()
            best_match = None
            
            for i in range(count):
                try:
                    info = pya.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        name = info.get('name', '')
                        # Simple case-insensitive check
                        if self.input_device_name.lower() in name.lower() or name.lower() in self.input_device_name.lower():
                             print(f"   Candidate {i}: {name}")
                             # Prioritize exact match or very close match if possible, but first match is okay for now
                             resolved_input_device_index = i
                             best_match = name
                             break
                except Exception:
                    continue
            
            if resolved_input_device_index is not None:
                print(f"[ADA] Resolved input device '{self.input_device_name}' to index {resolved_input_device_index} ({best_match})")
            else:
                print(f"[ADA] Could not find device matching '{self.input_device_name}'. Checking index...")

        # Fallback to index if Name lookup failed or wasn't provided
        if resolved_input_device_index is None and self.input_device_index is not None:
            try:
                candidate_index = int(self.input_device_index)
                info = pya.get_device_info_by_index(candidate_index)
                if info.get("maxInputChannels", 0) > 0:
                    resolved_input_device_index = candidate_index
                    print(f"[ADA] Using Input Device Index: {candidate_index} ({info.get('name', 'Unknown')})")
                else:
                    print(f"[ADA] Device index {candidate_index} has no input channels. Using default.")
            except Exception as e:
                print(f"[ADA] Invalid device index '{self.input_device_index}', using default. ({e})")

        if resolved_input_device_index is None:
             print("[ADA] Using Default Input Device")

        try:
            self.audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=resolved_input_device_index if resolved_input_device_index is not None else mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
        except OSError as e:
            print(f"[ADA] [ERR] Failed to open audio input stream: {e}")
            print("[ADA] [WARN] Audio features will be disabled. Please check microphone permissions.")
            if self.on_error:
                self.on_error(f"Failed to open microphone input: {e}. Check device selection and OS mic permissions.")
            return

        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        
        # VAD Constants
        VAD_THRESHOLD = 100  # RMS threshold — lower = more sensitive.
        SILENCE_DURATION = 1.2  # Seconds of silence before triggering STT
        _rms_log_counter = 0  # For periodic debug logging
        _is_processing_since = None  # Safety timeout for is_processing flag
        _startup_time = time.time()  # Used for intensive startup debug window
        
        print("[ADA] ✅ listen_audio() started — VAD is ACTIVE. Speak now to test.")
        print(f"[ADA] VAD threshold: {VAD_THRESHOLD} | Silence duration: {SILENCE_DURATION}s")
        
        while True:
            if self.paused:
                await asyncio.sleep(0.1)
                continue
                
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                
                # Safety: Reset is_processing if it's been stuck for >30 seconds
                if self.is_processing and _is_processing_since is not None:
                    if time.time() - _is_processing_since > 30.0:
                        print("[ADA DEBUG] [VAD] WARNING: is_processing was stuck for 30s — force-resetting.")
                        self.is_processing = False
                        _is_processing_since = None
                
                # Prevent Self-Interruption (Mute mic while ADA speaks)
                # We MUST read the data first to flush the OS buffer, then discard it.
                try:
                    ada_speaking = pygame.mixer.get_init() and pygame.mixer.music.get_busy()
                except Exception:
                    ada_speaking = False
                
                if ada_speaking:
                    if self.audio_buffer:
                        self.audio_buffer.clear()
                    self._is_speaking = False
                    self._silence_start_time = None
                    continue
                
                # 1. Send Audio (Gemini Live mode only)
                if self.out_queue:
                    await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
                
                # 2. Compute RMS for VAD
                count = len(data) // 2
                if count > 0:
                    shorts = struct.unpack(f"<{count}h", data)
                    sum_squares = sum(s**2 for s in shorts)
                    rms = int(math.sqrt(sum_squares / count))
                else:
                    rms = 0
                
                # Intensive debug for first 10 seconds: log every chunk
                _rms_log_counter += 1
                elapsed = time.time() - _startup_time
                if elapsed < 10.0:
                    # Log every 20 chunks in debug window (~1.3s intervals)
                    if _rms_log_counter % 20 == 0:
                        print(f"[ADA DEBUG] [VAD] RMS={rms} | threshold={VAD_THRESHOLD} | speaking={self._is_speaking} | paused={self.paused} | elapsed={elapsed:.1f}s")
                else:
                    # Normal operation: log every 50 chunks (~3s)
                    if _rms_log_counter % 50 == 0:
                        print(f"[ADA DEBUG] [VAD] Ambient RMS: {rms} (threshold: {VAD_THRESHOLD}, speaking: {self._is_speaking})")
                
                if rms > VAD_THRESHOLD:
                    # Speech Detected
                    self._silence_start_time = None
                    
                    if not self._is_speaking:
                        # NEW Speech Utterance Started
                        self._is_speaking = True
                        print(f"[ADA DEBUG] [VAD] Speech START detected (RMS: {rms}). Recording...")
                        
                        # Send ONE video frame when speech begins
                        if self._latest_image_payload and self.out_queue:
                            await self.out_queue.put(self._latest_image_payload)
                            
                else:
                    # Silence
                    if self._is_speaking:
                        if self._silence_start_time is None:
                            self._silence_start_time = time.time()
                        
                        elif time.time() - self._silence_start_time > SILENCE_DURATION:
                            # Silence confirmed — end of utterance
                            buffer_len = len(self.audio_buffer)
                            print(f"[ADA DEBUG] [VAD] Speech END detected. Buffer: {buffer_len} chunks ({buffer_len * CHUNK_SIZE * 2 // 1000}ms of audio).")
                            self._is_speaking = False
                            self._silence_start_time = None
                            
                            # TRIGGER PIPELINE TURN
                            if USE_PIPELINE and self.audio_buffer and not self.is_processing:
                                # Snapshot & clear buffer atomically before handing off
                                audio_snapshot = list(self.audio_buffer)
                                self.audio_buffer.clear()
                                _is_processing_since = time.time()
                                asyncio.create_task(self.process_pipeline_turn(audio_data=audio_snapshot))
                            elif self.is_processing:
                                print("[ADA DEBUG] [VAD] Skipping pipeline — already processing previous turn.")
                                self.audio_buffer.clear()  # Drop this utterance
                            elif not self.audio_buffer:
                                print("[ADA DEBUG] [VAD] Skipping pipeline — audio buffer is empty.")

                # Accumulate for Pipeline (only while actively speaking)
                if USE_PIPELINE and self._is_speaking:
                    self.audio_buffer.append(data)

            except Exception as e:
                print(f"[ADA DEBUG] [VAD] Error reading audio: {e}")
                traceback.print_exc()
                await asyncio.sleep(0.1)

    async def handle_cad_request(self, prompt):
        print(f"[ADA DEBUG] [CAD] Background Task Started: handle_cad_request('{prompt}')")
        if self.on_cad_status:
            self.on_cad_status("generating")
            
        # Auto-create project if stuck in temp
        if self.project_manager.current_project == "temp":
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_project_name = f"Project_{timestamp}"
            print(f"[ADA DEBUG] [CAD] Auto-creating project: {new_project_name}")
            
            success, msg = self.project_manager.create_project(new_project_name)
            if success:
                self.project_manager.switch_project(new_project_name)
                # Notify User (Optional, or rely on update)
                try:
                    await self.session.send(input=f"System Notification: Automatic Project Creation. Switched to new project '{new_project_name}'.", end_of_turn=False)
                    if self.on_project_update:
                         self.on_project_update(new_project_name)
                except Exception as e:
                    print(f"[ADA DEBUG] [ERR] Failed to notify auto-project: {e}")

        # Get project cad folder path
        cad_output_dir = str(self.project_manager.get_current_project_path() / "cad")
        
        # Call the secondary agent with project path
        cad_data = await self.cad_agent.generate_prototype(prompt, output_dir=cad_output_dir)
        
        if cad_data:
            print(f"[ADA DEBUG] [OK] CadAgent returned data successfully.")
            print(f"[ADA DEBUG] [INFO] Data Check: {len(cad_data.get('vertices', []))} vertices, {len(cad_data.get('edges', []))} edges.")
            
            if self.on_cad_data:
                print(f"[ADA DEBUG] [SEND] Dispatching data to frontend callback...")
                self.on_cad_data(cad_data)
                print(f"[ADA DEBUG] [SENT] Dispatch complete.")
            
            # Save to Project
            if 'file_path' in cad_data:
                self.project_manager.save_cad_artifact(cad_data['file_path'], prompt)
            else:
                 # Fallback (legacy support)
                 self.project_manager.save_cad_artifact("output.stl", prompt)

            # Notify the model that the task is done - this triggers speech about completion
            completion_msg = "System Notification: CAD generation is complete! The 3D model is now displayed for the user. Let them know it's ready."
            try:
                await self.session.send(input=completion_msg, end_of_turn=True)
                print(f"[ADA DEBUG] [NOTE] Sent completion notification to model.")
            except Exception as e:
                 print(f"[ADA DEBUG] [ERR] Failed to send completion notification: {e}")

        else:
            print(f"[ADA DEBUG] [ERR] CadAgent returned None.")
            # Optionally notify failure
            try:
                await self.session.send(input="System Notification: CAD generation failed.", end_of_turn=True)
            except Exception:
                pass



    async def handle_write_file(self, path, content):
        print(f"[ADA DEBUG] [FS] Writing file: '{path}'")
        
        # Auto-create project if stuck in temp
        if self.project_manager.current_project == "temp":
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_project_name = f"Project_{timestamp}"
            print(f"[ADA DEBUG] [FS] Auto-creating project: {new_project_name}")
            
            success, msg = self.project_manager.create_project(new_project_name)
            if success:
                self.project_manager.switch_project(new_project_name)
                # Notify User
                try:
                    await self.session.send(input=f"System Notification: Automatic Project Creation. Switched to new project '{new_project_name}'.", end_of_turn=False)
                    if self.on_project_update:
                         self.on_project_update(new_project_name)
                except Exception as e:
                    print(f"[ADA DEBUG] [ERR] Failed to notify auto-project: {e}")
        
        # Force path to be relative to current project
        # If absolute path is provided, we try to strip it or just ignore it and use basename
        filename = os.path.basename(path)
        
        # If path contained subdirectories (e.g. "backend/server.py"), preserving that structure might be desired IF it's within the project.
        # But for safety, and per user request to "always create the file in the project", 
        # we will root it in the current project path.
        
        current_project_path = self.project_manager.get_current_project_path()
        final_path = current_project_path / filename # Simple flat structure for now, or allow relative?
        
        # If the user specifically wanted a subfolder, they might have provided "sub/file.txt".
        # Let's support relative paths if they don't start with /
        if not os.path.isabs(path):
             final_path = current_project_path / path
        
        print(f"[ADA DEBUG] [FS] Resolved path: '{final_path}'")

        try:
            # Ensure parent exists
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            with open(final_path, 'w', encoding='utf-8') as f:
                f.write(content)
            result = f"File '{final_path.name}' written successfully to project '{self.project_manager.current_project}'."
        except Exception as e:
            result = f"Failed to write file '{path}': {str(e)}"

        print(f"[ADA DEBUG] [FS] Result: {result}")
        try:
             await self.session.send(input=f"System Notification: {result}", end_of_turn=True)
        except Exception as e:
             print(f"[ADA DEBUG] [ERR] Failed to send fs result: {e}")

    async def handle_read_directory(self, path):
        print(f"[ADA DEBUG] [FS] Reading directory: '{path}'")
        try:
            if not os.path.exists(path):
                result = f"Directory '{path}' does not exist."
            else:
                items = os.listdir(path)
                result = f"Contents of '{path}': {', '.join(items)}"
        except Exception as e:
            result = f"Failed to read directory '{path}': {str(e)}"

        print(f"[ADA DEBUG] [FS] Result: {result}")
        try:
             await self.session.send(input=f"System Notification: {result}", end_of_turn=True)
        except Exception as e:
             print(f"[ADA DEBUG] [ERR] Failed to send fs result: {e}")

    async def handle_read_file(self, path):
        print(f"[ADA DEBUG] [FS] Reading file: '{path}'")
        try:
            if not os.path.exists(path):
                result = f"File '{path}' does not exist."
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                result = f"Content of '{path}':\n{content}"
        except Exception as e:
            result = f"Failed to read file '{path}': {str(e)}"

        print(f"[ADA DEBUG] [FS] Result: {result}")
        try:
             await self.session.send(input=f"System Notification: {result}", end_of_turn=True)
        except Exception as e:
             print(f"[ADA DEBUG] [ERR] Failed to send fs result: {e}")

    async def handle_web_agent_request(self, prompt):
        print(f"[ADA DEBUG] [WEB] Web Agent Task: '{prompt}'")
        
        async def update_frontend(image_b64, log_text):
            if self.on_web_data:
                 self.on_web_data({"image": image_b64, "log": log_text})
                 
        # Run the web agent and wait for it to return
        result = await self.web_agent.run_task(prompt, update_callback=update_frontend)
        print(f"[ADA DEBUG] [WEB] Web Agent Task Returned: {result}")
        
        # Send the final result back to the main model
        try:
             await self.session.send(input=f"System Notification: Web Agent has finished.\nResult: {result}", end_of_turn=True)
        except Exception as e:
             print(f"[ADA DEBUG] [ERR] Failed to send web agent result to model: {e}")

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        try:
            while True:
                turn = self.session.receive()
                async for response in turn:
                    # 1. Handle Audio Data
                    if data := response.data:
                        print(f"[ADA DEBUG] [AUDIO] Received {len(data)} bytes of audio from model.")
                        self.audio_in_queue.put_nowait(data)

                    # 2. Handle Transcription (User & Model)
                    if response.server_content:
                        if response.server_content.input_transcription:
                            transcript = response.server_content.input_transcription.text
                            if transcript:
                                # Skip if this is an exact duplicate event
                                if transcript != self._last_input_transcription:
                                    # Calculate delta (Gemini may send cumulative or chunk-based text)
                                    delta = transcript
                                    if transcript.startswith(self._last_input_transcription):
                                        delta = transcript[len(self._last_input_transcription):]
                                    self._last_input_transcription = transcript
                                    
                                    # Only send if there's new text
                                    if delta:
                                        # User is speaking, so interrupt model playback!
                                        # self.clear_audio_queue() # TEMP: Disable for debugging silence

                                        # Send to frontend (Streaming)
                                        if self.on_transcription:
                                             self.on_transcription({"sender": "User", "text": delta})
                                        
                                        # Buffer for Logging
                                        if self.chat_buffer["sender"] != "User":
                                            # Flush previous if exists
                                            if self.chat_buffer["sender"] and self.chat_buffer["text"].strip():
                                                self.project_manager.log_chat(self.chat_buffer["sender"], self.chat_buffer["text"])
                                            # Start new
                                            self.chat_buffer = {"sender": "User", "text": delta}
                                        else:
                                            # Append
                                            self.chat_buffer["text"] += delta
                        
                        if response.server_content.output_transcription:
                            transcript = response.server_content.output_transcription.text
                            if transcript:
                                # Skip if this is an exact duplicate event
                                if transcript != self._last_output_transcription:
                                    # Calculate delta (Gemini may send cumulative or chunk-based text)
                                    delta = transcript
                                    if transcript.startswith(self._last_output_transcription):
                                        delta = transcript[len(self._last_output_transcription):]
                                    self._last_output_transcription = transcript
                                    
                                    # Only send if there's new text
                                    if delta:
                                        # Send to frontend (Streaming)
                                        if self.on_transcription:
                                             self.on_transcription({"sender": "ADA", "text": delta})
                                        
                                        # Buffer for Logging
                                        if self.chat_buffer["sender"] != "ADA":
                                            # Flush previous
                                            if self.chat_buffer["sender"] and self.chat_buffer["text"].strip():
                                                self.project_manager.log_chat(self.chat_buffer["sender"], self.chat_buffer["text"])
                                            # Start new
                                            self.chat_buffer = {"sender": "ADA", "text": delta}
                                        else:
                                            # Append
                                            self.chat_buffer["text"] += delta
                        
                        # Flush buffer on turn completion if needed, 
                        # but usually better to wait for sender switch or explicit end.
                        # We can also check turn_complete signal if available in response.server_content.model_turn etc

                    # 3. Handle Tool Calls
                    if response.tool_call:
                        print("The tool was called")
                        function_responses = []
                        for fc in response.tool_call.function_calls:
                            if fc.name in ["generate_cad", "run_web_agent", "write_file", "read_directory", "read_file", "create_project", "switch_project", "list_projects", "list_smart_devices", "control_light", "discover_printers", "print_stl", "get_print_status", "iterate_cad"]:
                                prompt = fc.args.get("prompt", "") # Prompt is not present for all tools
                                
                                # Check Permissions (Default to True if not set)
                                confirmation_required = self.permissions.get(fc.name, True)
                                
                                if not confirmation_required:
                                    print(f"[ADA DEBUG] [TOOL] Permission check: '{fc.name}' -> AUTO-ALLOW")
                                    # Skip confirmation block and jump to execution
                                    pass
                                else:
                                    # Confirmation Logic
                                    if self.on_tool_confirmation:
                                        import uuid
                                        request_id = str(uuid.uuid4())
                                    print(f"[ADA DEBUG] [STOP] Requesting confirmation for '{fc.name}' (ID: {request_id})")
                                    
                                    future = asyncio.Future()
                                    self._pending_confirmations[request_id] = future
                                    
                                    self.on_tool_confirmation({
                                        "id": request_id, 
                                        "tool": fc.name, 
                                        "args": fc.args
                                    })
                                    
                                    try:
                                        # Wait for user response
                                        confirmed = await future

                                    finally:
                                        self._pending_confirmations.pop(request_id, None)

                                    print(f"[ADA DEBUG] [CONFIRM] Request {request_id} resolved. Confirmed: {confirmed}")

                                    if not confirmed:
                                        print(f"[ADA DEBUG] [DENY] Tool call '{fc.name}' denied by user.")
                                        function_response = types.FunctionResponse(
                                            id=fc.id,
                                            name=fc.name,
                                            response={
                                                "result": "User denied the request to use this tool.",
                                            }
                                        )
                                        function_responses.append(function_response)
                                        continue

                                    if not confirmed:
                                        print(f"[ADA DEBUG] [DENY] Tool call '{fc.name}' denied by user.")
                                        function_response = types.FunctionResponse(
                                            id=fc.id,
                                            name=fc.name,
                                            response={
                                                "result": "User denied the request to use this tool.",
                                            }
                                        )
                                        function_responses.append(function_response)
                                        continue

                                # If confirmed (or no callback configured, or auto-allowed), proceed
                                if fc.name == "generate_cad":
                                    print(f"\n[ADA DEBUG] --------------------------------------------------")
                                    print(f"[ADA DEBUG] [TOOL] Tool Call Detected: 'generate_cad'")
                                    print(f"[ADA DEBUG] [IN] Arguments: prompt='{prompt}'")
                                    
                                    asyncio.create_task(self.handle_cad_request(prompt))
                                    # No function response needed - model already acknowledged when user asked
                                
                                elif fc.name == "run_web_agent":
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'run_web_agent' with prompt='{prompt}'")
                                    asyncio.create_task(self.handle_web_agent_request(prompt))
                                    
                                    result_text = "Web Navigation started. Do not reply to this message."
                                    function_response = types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response={
                                            "result": result_text,
                                        }
                                    )
                                    print(f"[ADA DEBUG] [RESPONSE] Sending function response: {function_response}")
                                    function_responses.append(function_response)



                                elif fc.name == "write_file":
                                    path = fc.args["path"]
                                    content = fc.args["content"]
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'write_file' path='{path}'")
                                    asyncio.create_task(self.handle_write_file(path, content))
                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": "Writing file..."}
                                    )
                                    function_responses.append(function_response)

                                elif fc.name == "read_directory":
                                    path = fc.args["path"]
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'read_directory' path='{path}'")
                                    asyncio.create_task(self.handle_read_directory(path))
                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": "Reading directory..."}
                                    )
                                    function_responses.append(function_response)

                                elif fc.name == "read_file":
                                    path = fc.args["path"]
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'read_file' path='{path}'")
                                    asyncio.create_task(self.handle_read_file(path))
                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": "Reading file..."}
                                    )
                                    function_responses.append(function_response)

                                elif fc.name == "create_project":
                                    name = fc.args["name"]
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'create_project' name='{name}'")
                                    success, msg = self.project_manager.create_project(name)
                                    if success:
                                        # Auto-switch to the newly created project
                                        self.project_manager.switch_project(name)
                                        msg += f" Switched to '{name}'."
                                        if self.on_project_update:
                                            self.on_project_update(name)
                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": msg}
                                    )
                                    function_responses.append(function_response)

                                elif fc.name == "switch_project":
                                    name = fc.args["name"]
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'switch_project' name='{name}'")
                                    success, msg = self.project_manager.switch_project(name)
                                    if success:
                                        if self.on_project_update:
                                            self.on_project_update(name)
                                        # Gather project context and send to AI (silently, no response expected)
                                        context = self.project_manager.get_project_context()
                                        print(f"[ADA DEBUG] [PROJECT] Sending project context to AI ({len(context)} chars)")
                                        try:
                                            await self.session.send(input=f"System Notification: {msg}\n\n{context}", end_of_turn=False)
                                        except Exception as e:
                                            print(f"[ADA DEBUG] [ERR] Failed to send project context: {e}")
                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": msg}
                                    )
                                    function_responses.append(function_response)
                                
                                elif fc.name == "list_projects":
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'list_projects'")
                                    projects = self.project_manager.list_projects()
                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": f"Available projects: {', '.join(projects)}"}
                                    )
                                    function_responses.append(function_response)

                                elif fc.name == "list_smart_devices":
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'list_smart_devices'")
                                    # Use cached devices directly for speed
                                    # devices_dict is {ip: SmartDevice}
                                    
                                    dev_summaries = []
                                    frontend_list = []
                                    
                                    for ip, d in self.kasa_agent.devices.items():
                                        dev_type = "unknown"
                                        if d.is_bulb: dev_type = "bulb"
                                        elif d.is_plug: dev_type = "plug"
                                        elif d.is_strip: dev_type = "strip"
                                        elif d.is_dimmer: dev_type = "dimmer"
                                        
                                        # Format for Model
                                        info = f"{d.alias} (IP: {ip}, Type: {dev_type})"
                                        if d.is_on:
                                            info += " [ON]"
                                        else:
                                            info += " [OFF]"
                                        dev_summaries.append(info)
                                        
                                        # Format for Frontend
                                        frontend_list.append({
                                            "ip": ip,
                                            "alias": d.alias,
                                            "model": d.model,
                                            "type": dev_type,
                                            "is_on": d.is_on,
                                            "brightness": d.brightness if d.is_bulb or d.is_dimmer else None,
                                            "hsv": d.hsv if d.is_bulb and d.is_color else None,
                                            "has_color": d.is_color if d.is_bulb else False,
                                            "has_brightness": d.is_dimmable if d.is_bulb or d.is_dimmer else False
                                        })
                                    
                                    result_str = "No devices found in cache."
                                    if dev_summaries:
                                        result_str = "Found Devices (Cached):\n" + "\n".join(dev_summaries)
                                    
                                    # Trigger frontend update
                                    if self.on_device_update:
                                        self.on_device_update(frontend_list)

                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": result_str}
                                    )
                                    function_responses.append(function_response)

                                elif fc.name == "control_light":
                                    target = fc.args["target"]
                                    action = fc.args["action"]
                                    brightness = fc.args.get("brightness")
                                    color = fc.args.get("color")
                                    
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'control_light' Target='{target}' Action='{action}'")
                                    
                                    result_msg = f"Action '{action}' on '{target}' failed."
                                    success = False
                                    
                                    if action == "turn_on":
                                        success = await self.kasa_agent.turn_on(target)
                                        if success:
                                            result_msg = f"Turned ON '{target}'."
                                    elif action == "turn_off":
                                        success = await self.kasa_agent.turn_off(target)
                                        if success:
                                            result_msg = f"Turned OFF '{target}'."
                                    elif action == "set":
                                        success = True
                                        result_msg = f"Updated '{target}':"
                                    
                                    # Apply extra attributes if 'set' or if we just turned it on and want to set them too
                                    if success or action == "set":
                                        if brightness is not None:
                                            sb = await self.kasa_agent.set_brightness(target, brightness)
                                            if sb:
                                                result_msg += f" Set brightness to {brightness}."
                                        if color is not None:
                                            sc = await self.kasa_agent.set_color(target, color)
                                            if sc:
                                                result_msg += f" Set color to {color}."

                                    # Notify Frontend of State Change
                                    if success:
                                        # We don't need full discovery, just refresh known state or push update
                                        # But for simplicity, let's get the standard list representation
                                        # KasaAgent updates its internal state on control, so we can rebuild the list
                                        
                                        # Quick rebuild of list from internal dict
                                        updated_list = []
                                        for ip, dev in self.kasa_agent.devices.items():
                                            # We need to ensure we have the correct dict structure expected by frontend
                                            # We duplicate logic from KasaAgent.discover_devices a bit, but that's okay for now or we can add a helper
                                            # Ideally KasaAgent has a 'get_devices_list()' method.
                                            # Use the cached objects in self.kasa_agent.devices
                                            
                                            dev_type = "unknown"
                                            if dev.is_bulb: dev_type = "bulb"
                                            elif dev.is_plug: dev_type = "plug"
                                            elif dev.is_strip: dev_type = "strip"
                                            elif dev.is_dimmer: dev_type = "dimmer"

                                            d_info = {
                                                "ip": ip,
                                                "alias": dev.alias,
                                                "model": dev.model,
                                                "type": dev_type,
                                                "is_on": dev.is_on,
                                                "brightness": dev.brightness if dev.is_bulb or dev.is_dimmer else None,
                                                "hsv": dev.hsv if dev.is_bulb and dev.is_color else None,
                                                "has_color": dev.is_color if dev.is_bulb else False,
                                                "has_brightness": dev.is_dimmable if dev.is_bulb or dev.is_dimmer else False
                                            }
                                            updated_list.append(d_info)
                                            
                                        if self.on_device_update:
                                            self.on_device_update(updated_list)
                                    else:
                                        # Report Error
                                        if self.on_error:
                                            self.on_error(result_msg)

                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": result_msg}
                                    )
                                    function_responses.append(function_response)

                                elif fc.name == "discover_printers":
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'discover_printers'")
                                    printers = await self.printer_agent.discover_printers()
                                    # Format for model
                                    if printers:
                                        printer_list = []
                                        for p in printers:
                                            printer_list.append(f"{p['name']} ({p['host']}:{p['port']}, type: {p['printer_type']})")
                                        result_str = "Found Printers:\n" + "\n".join(printer_list)
                                    else:
                                        result_str = "No printers found on network. Ensure printers are on and running OctoPrint/Moonraker."
                                    
                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": result_str}
                                    )
                                    function_responses.append(function_response)

                                elif fc.name == "print_stl":
                                    stl_path = fc.args["stl_path"]
                                    printer = fc.args["printer"]
                                    profile = fc.args.get("profile")
                                    
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'print_stl' STL='{stl_path}' Printer='{printer}'")
                                    
                                    # Resolve 'current' to project STL
                                    if stl_path.lower() == "current":
                                        stl_path = "output.stl" # Let printer agent resolve it in root_path

                                    # Get current project path
                                    project_path = str(self.project_manager.get_current_project_path())
                                    
                                    result = await self.printer_agent.print_stl(
                                        stl_path, 
                                        printer, 
                                        profile, 
                                        root_path=project_path
                                    )
                                    result_str = result.get("message", "Unknown result")
                                    
                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": result_str}
                                    )
                                    function_responses.append(function_response)

                                elif fc.name == "get_print_status":
                                    printer = fc.args["printer"]
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'get_print_status' Printer='{printer}'")
                                    
                                    status = await self.printer_agent.get_print_status(printer)
                                    if status:
                                        result_str = f"Printer: {status.printer}\n"
                                        result_str += f"State: {status.state}\n"
                                        result_str += f"Progress: {status.progress_percent:.1f}%\n"
                                        if status.time_remaining:
                                            result_str += f"Time Remaining: {status.time_remaining}\n"
                                        if status.time_elapsed:
                                            result_str += f"Time Elapsed: {status.time_elapsed}\n"
                                        if status.filename:
                                            result_str += f"File: {status.filename}\n"
                                        if status.temperatures:
                                            temps = status.temperatures
                                            if "hotend" in temps:
                                                result_str += f"Hotend: {temps['hotend']['current']:.0f}°C / {temps['hotend']['target']:.0f}°C\n"
                                            if "bed" in temps:
                                                result_str += f"Bed: {temps['bed']['current']:.0f}°C / {temps['bed']['target']:.0f}°C"
                                    else:
                                        result_str = f"Could not get status for printer '{printer}'. Ensure it is discovered first."
                                    
                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": result_str}
                                    )
                                    function_responses.append(function_response)

                                elif fc.name == "iterate_cad":
                                    prompt = fc.args["prompt"]
                                    print(f"[ADA DEBUG] [TOOL] Tool Call: 'iterate_cad' Prompt='{prompt}'")
                                    
                                    # Emit status
                                    if self.on_cad_status:
                                        self.on_cad_status("generating")
                                    
                                    # Get project cad folder path
                                    cad_output_dir = str(self.project_manager.get_current_project_path() / "cad")
                                    
                                    # Call CadAgent to iterate on the design
                                    cad_data = await self.cad_agent.iterate_prototype(prompt, output_dir=cad_output_dir)
                                    
                                    if cad_data:
                                        print(f"[ADA DEBUG] [OK] CadAgent iteration returned data successfully.")
                                        
                                        # Dispatch to frontend
                                        if self.on_cad_data:
                                            print(f"[ADA DEBUG] [SEND] Dispatching iterated CAD data to frontend...")
                                            self.on_cad_data(cad_data)
                                            print(f"[ADA DEBUG] [SENT] Dispatch complete.")
                                        
                                        # Save to Project
                                        self.project_manager.save_cad_artifact("output.stl", f"Iteration: {prompt}")
                                        
                                        result_str = f"Successfully iterated design: {prompt}. The updated 3D model is now displayed."
                                    else:
                                        print(f"[ADA DEBUG] [ERR] CadAgent iteration returned None.")
                                        result_str = f"Failed to iterate design with prompt: {prompt}"
                                    
                                    function_response = types.FunctionResponse(
                                        id=fc.id, name=fc.name, response={"result": result_str}
                                    )
                                    function_responses.append(function_response)
                        if function_responses:
                            await self.session.send_tool_response(function_responses=function_responses)
                
                # Turn/Response Loop Finished
                self.flush_chat()

                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()
        except Exception as e:
            print(f"Error in receive_audio: {e}")
            traceback.print_exc()
            # CRITICAL: Re-raise to crash the TaskGroup and trigger outer loop reconnect
            raise e

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
            output_device_index=self.output_device_index,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            if bytestream is None: continue
            
            if self.on_audio_data:
                self.on_audio_data(bytestream)
            
            # Simple heartbeat log (every 20 chunks to avoid spam)
            if not hasattr(self, '_play_count'): self._play_count = 0
            self._play_count += 1
            if self._play_count % 20 == 0:
                print(f"[ADA DEBUG] [AUDIO] Successfully writing chunk {self._play_count} to speaker...")
                
            await asyncio.to_thread(stream.write, bytestream)

    async def get_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0, cv2.CAP_AVFOUNDATION)
        while True:
            if self.paused:
                await asyncio.sleep(0.1)
                continue
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            if self.out_queue:
                await self.out_queue.put(frame)
        cap.release()

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        image_bytes = image_io.read()
        return {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()}

    async def _get_screen(self):
        pass 
    async def get_screen(self):
         pass

    async def _execute_pipeline_tool(self, tool_name, args, manage_focus=True):
        """Execute a tool call from the pipeline (prompt-based tool calling)."""
        import pyautogui

        focus_active = manage_focus and tool_name in FOCUS_TOOL_NAMES
        if focus_active:
            await self._set_focus_mode(True)

        try:
            if tool_name == "create_project":
                s, result = self.project_manager.create_project(args.get("name", "temp"))
                return result

            elif tool_name == "open_application":
                app_name = args.get("app_name", "").lower().strip()
                url = (args.get("url") or "").strip()
                query = (args.get("query") or "").strip()
                try:
                    import re
                    if "spotify" in app_name:
                        os.startfile("spotify:")
                    elif "chrome" in app_name or "browser" in app_name:
                        target_url = ""
                        opened_in_chrome = False
                        if query:
                            import urllib.parse
                            encoded_query = urllib.parse.quote(query)
                            target_url = f"https://www.google.com/search?q={encoded_query}"
                        elif url:
                            if not re.match(r"^https?://", url, re.IGNORECASE):
                                target_url = "https://" + url
                            else:
                                target_url = url

                        chrome_exe = self._find_chrome_executable()
                        if chrome_exe:
                            import subprocess
                            if target_url:
                                subprocess.Popen([chrome_exe, target_url])
                            else:
                                subprocess.Popen([chrome_exe])
                            opened_in_chrome = True
                        elif target_url:
                            exit_code = os.system(f"start chrome \"{target_url}\"")
                            if exit_code != 0:
                                os.startfile(target_url)
                            else:
                                opened_in_chrome = True
                        else:
                            os.startfile("http://www.google.com")
                    else:
                        os.system(f"start {app_name}")

                    # Wait for the application to fully load before next tool call
                    wait_sec = args.get("wait", 5)
                    if wait_sec:
                        await asyncio.sleep(wait_sec)
                    if "spotify" in app_name:
                        await self._focus_window_by_title(["spotify"], timeout=6.0)
                    elif "chrome" in app_name or "browser" in app_name:
                        await self._focus_window_by_title(["chrome", "google chrome"], timeout=6.0)

                    browser_label = (
                        "Chrome"
                        if ("chrome" in app_name and opened_in_chrome)
                        else ("default browser" if "chrome" in app_name else "browser")
                    )
                    if query:
                        return f"Opened {browser_label} and searched for '{query}'."
                    if url:
                        return f"Opened {browser_label} to '{url}'."
                    return f"Successfully opened {app_name}."
                except Exception as e:
                    return f"Failed to open {app_name}: {e}"

            elif tool_name == "terminate_application":
                app_name = (args.get("app_name") or "").strip()
                force = args.get("force", True)
                if not app_name:
                    return "No app name provided for termination."

                app_key = app_name.lower().replace(".exe", "")
                known = {
                    "spotify": "Spotify.exe",
                    "chrome": "chrome.exe",
                    "google chrome": "chrome.exe",
                    "edge": "msedge.exe",
                    "msedge": "msedge.exe",
                    "firefox": "firefox.exe",
                }
                process_name = known.get(app_key, app_name)
                if not process_name.lower().endswith(".exe") and sys.platform == "win32":
                    process_name += ".exe"

                try:
                    if sys.platform == "win32":
                        import subprocess
                        cmd = ["taskkill", "/im", process_name]
                        if force:
                            cmd.insert(1, "/f")
                        completed = subprocess.run(cmd, capture_output=True, text=True)
                        if completed.returncode == 0:
                            return f"Closed {process_name}."
                        detail = completed.stderr.strip() or completed.stdout.strip()
                        return f"Failed to close {process_name}: {detail}"
                    else:
                        import subprocess
                        cmd = ["pkill"] + (["-9"] if force else []) + ["-f", process_name]
                        completed = subprocess.run(cmd, capture_output=True, text=True)
                        if completed.returncode == 0:
                            return f"Closed {process_name}."
                        detail = completed.stderr.strip() or completed.stdout.strip()
                        return f"Failed to close {process_name}: {detail}"
                except Exception as e:
                    return f"Failed to close {process_name}: {e}"

            elif tool_name == "play_music":
                query = args.get("query", "").strip()
                if not query:
                    return "No song or artist provided for playback."
                try:
                    import urllib.parse
                    encoded_query = urllib.parse.quote(query)
                    os.startfile(f"spotify:search:{encoded_query}")
                    await asyncio.sleep(4.0) # Wait for search results to populate
                    await self._focus_window_by_title(["spotify"], timeout=8.0)
                    await asyncio.sleep(1.0)

                    # Try multiple labels for the play button with double-click for reliability
                    play_labels = [
                        "the large green play button next to the search result",
                        "the circular play button with a black triangle",
                        "the main play button at the bottom of the screen",
                        "green play button"
                    ]
                    clicked = False
                    for label in play_labels:
                        print(f"[ADA DEBUG] [TOOL] Attempting to find and click '{label}'...")
                        point = await self._point_ui_element(label)
                        if point and point[0] is not None:
                            screen_w, screen_h = pyautogui.size()
                            target_x = int(point[0] * screen_w)
                            target_y = int(point[1] * screen_h)
                            
                            print(f"[ADA DEBUG] [TOOL] Clicking at {target_x}, {target_y} for '{label}'")
                            pyautogui.moveTo(target_x, target_y, duration=0.5)
                            pyautogui.doubleClick(target_x, target_y)
                            clicked = True
                            await asyncio.sleep(1.0)
                            break
                    
                    if not clicked:
                        return (
                            "I searched for the song but could not find a clear Play button to click. "
                            "Please ensure Spotify is visible on your primary monitor."
                        )
                    return f"Successfully sent play command for '{query}' on Spotify (Double-clicked play button)."
                except Exception as e:
                    return f"Failed to play music: {e}"

            elif tool_name == "see":
                prompt = args.get("prompt", "Describe this image in detail.")
                if self._latest_image_payload:
                    try:
                        img_data = self._latest_image_payload['data']
                        data_uri = f"data:image/jpeg;base64,{img_data}"
                        url = "https://api.moondream.ai/v1/query"
                        headers = {"X-Moondream-Auth": moondream_api_key}
                        payload = {"image_url": data_uri, "question": prompt}
                        async with httpx.AsyncClient() as v_client:
                            v_resp = await v_client.post(url, headers=headers, json=payload, timeout=20.0)
                        if v_resp.status_code == 200:
                            return v_resp.json().get("answer", "I see it, but I can't describe it.")
                        return f"Vision error: {v_resp.status_code} - {v_resp.text}"
                    except Exception as e:
                        return f"Vision system failure: {e}"
                return "I'm sorry, my camera is currently not providing a clear frame."

            elif tool_name == "see_desktop":
                prompt = args.get("prompt", "Describe the screen.")
                try:
                    with mss.mss() as sct:
                        monitor = sct.monitors[1]
                        screenshot = sct.grab(monitor)
                        img = PIL.Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=85)
                        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    data_uri = f"data:image/jpeg;base64,{img_data}"
                    url = "https://api.moondream.ai/v1/query"
                    headers = {"X-Moondream-Auth": moondream_api_key}
                    payload = {"image_url": data_uri, "question": f"Based on this screenshot, {prompt}"}
                    async with httpx.AsyncClient() as v_client:
                        v_resp = await v_client.post(url, headers=headers, json=payload, timeout=30.0)
                    if v_resp.status_code == 200:
                        return v_resp.json().get("answer", "I see the screen but can't describe it.")
                    return f"Desktop Vision error: {v_resp.status_code}"
                except Exception as e:
                    return f"Desktop Vision failure: {e}"

            elif tool_name == "get_ui_coordinates":
                object_name = args.get("object_name")
                try:
                    with mss.mss() as sct:
                        monitor = sct.monitors[1]
                        screenshot = sct.grab(monitor)
                        img = PIL.Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=85)
                        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    data_uri = f"data:image/jpeg;base64,{img_data}"
                    url = "https://api.moondream.ai/v1/point"
                    headers = {"X-Moondream-Auth": moondream_api_key}
                    payload = {"image_url": data_uri, "object": object_name}
                    async with httpx.AsyncClient() as v_client:
                        v_resp = await v_client.post(url, headers=headers, json=payload, timeout=30.0)
                    if v_resp.status_code == 200:
                        points = v_resp.json().get("points", [])
                        if points:
                            pt = points[0]
                            res_x = int(pt['x'] * 100)
                            res_y = int(pt['y'] * 100)
                            return f"Found '{object_name}' at {res_x}%, {res_y}%"
                        return f"Could not find '{object_name}' on screen."
                    return f"Pointing error: {v_resp.status_code}"
                except Exception as e:
                    return f"Pointing failure: {e}"

            elif tool_name == "mouse_action":
                action = args.get("action")
                x_pct = args.get("x", 0)
                y_pct = args.get("y", 0)
                try:
                    screen_w, screen_h = pyautogui.size()
                    target_x = int((x_pct / 100) * screen_w)
                    target_y = int((y_pct / 100) * screen_h)
                    if action == "move":
                        pyautogui.moveTo(target_x, target_y, duration=0.5)
                    elif action == "click":
                        pyautogui.click(target_x, target_y)
                    elif action == "double_click":
                        pyautogui.doubleClick(target_x, target_y)
                    elif action == "right_click":
                        pyautogui.rightClick(target_x, target_y)
                    return f"Performed {action} at {x_pct}%, {y_pct}% ({target_x}, {target_y})"
                except Exception as e:
                    return f"Mouse action failed: {e}"

            elif tool_name == "keyboard_action":
                action = args.get("action")
                text = args.get("text", "")
                try:
                    if action == "type":
                        pyautogui.write(text, interval=0.05)
                    elif action == "press":
                        keys = text.split('+')
                        if len(keys) > 1:
                            pyautogui.hotkey(*keys)
                        else:
                            pyautogui.press(text)
                    return f"Performed keyboard {action}: {text}"
                except Exception as e:
                    return f"Keyboard action failed: {e}"

            return f"Unknown tool: {tool_name}"
        finally:
            if focus_active:
                await self._set_focus_mode(False)

    def _strip_audio_tags(self, text):
        import re
        return re.sub(r'\[.*?\]', '', text).strip()

    async def _synthesize_deepgram(self, text):
        clean_text = self._strip_audio_tags(text)
        if not clean_text: return None
        
        try:
            print(f"[ADA DEBUG] [VOICE] Synthesizing with Deepgram Aura-2...")
            dg_url = f"https://api.deepgram.com/v1/speak?model=aura-2-asteria-en"
            dg_headers = {"Authorization": f"Token {deepgram_api_key}", "Content-Type": "application/json"}
            async with httpx.AsyncClient() as dg_client:
                dg_resp = await dg_client.post(dg_url, headers=dg_headers, json={"text": clean_text}, timeout=10.0)
            
            if dg_resp.status_code == 200:
                output_file = f"temp_ada_voice_{uuid.uuid4().hex[:8]}.mp3"
                with open(output_file, "wb") as f: f.write(dg_resp.content)
                return output_file
        except Exception as e:
            print(f"[ADA DEBUG] [ERR] Deepgram synthesis failed: {e}")
        return None

    async def _synthesize_elevenlabs(self, text):
        # Strip routing tag but KEEP emotional tags
        clean_text = text.replace("[VOICE: HD]", "").strip()
        
        if not elevenlabs_api_key or not elevenlabs_voice_id:
            print("[ADA DEBUG] [WARN] ElevenLabs credentials missing. Falling back to Deepgram...")
            return await self._synthesize_deepgram(clean_text)
            
        try:
            print(f"[ADA DEBUG] [VOICE] Synthesizing with ElevenLabs v3...")
            el_url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}"
            el_headers = {
                "xi-api-key": elevenlabs_api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "text": clean_text,
                "model_id": "eleven_v3",
                "voice_settings": {
                    "stability": 0.35,
                    "similarity_boost": 0.75,
                    "style": 0.45,
                    "use_speaker_boost": True
                }
            }
            async with httpx.AsyncClient() as el_client:
                el_resp = await el_client.post(el_url, headers=el_headers, json=payload, timeout=25.0)
            
            if el_resp.status_code == 200:
                print(f"[ADA SUCCESS] ElevenLabs v3 synthesis successful.")
                output_file = f"temp_ada_voice_{uuid.uuid4().hex[:8]}.mp3"
                with open(output_file, "wb") as f: f.write(el_resp.content)
                return output_file
            else:
                print(f"[ADA ERROR] ElevenLabs API error: {el_resp.status_code} - {el_resp.text}")
        except Exception as e:
            print(f"[ADA DEBUG] [ERR] ElevenLabs synthesis failed: {e}")
        
        return await self._synthesize_deepgram(clean_text)

    async def process_pipeline_turn(self, text_prompt=None, audio_data=None):
        """Pipeline Turn: User Audio/Text -> Brain (GPT-4o/Llama) -> Native Tools -> TTS"""
        if self.is_processing: return
        self.is_processing = True
        
        # Notify user that we are thinking
        if self.on_transcription:
            print(f"[ADA DEBUG] [PIPELINE] Started processing...")
        
        try:
            # 1. Input routing & Transcription
            if text_prompt:
                user_message = text_prompt
            else:
                if not self.stt_available:
                    self.is_processing = False
                    return
                raw_chunks = audio_data if audio_data else self.audio_buffer
                if not raw_chunks:
                    self.is_processing = False
                    return
                
                combined_audio = b"".join(raw_chunks)
                if not audio_data:
                    self.audio_buffer.clear()

                duration_ms = (len(combined_audio) * 1000) // 32000
                if duration_ms < 500:
                    self.is_processing = False
                    return

                import speech_recognition as sr
                import io
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(combined_audio)
                wav_buffer.seek(0)

                recognizer = sr.Recognizer()
                try:
                    with sr.AudioFile(wav_buffer) as source:
                        audio_recorded = recognizer.record(source)
                    user_message = await asyncio.to_thread(recognizer.recognize_google, audio_recorded, language="en-US")
                    user_message = user_message.strip()
                    if not user_message:
                        self.is_processing = False
                        return
                except Exception:
                    self.is_processing = False
                    return

                if self.on_transcription:
                    self.on_transcription({"sender": "User", "text": user_message})
                if self.project_manager:
                    self.project_manager.log_chat("User", user_message)

            # 2. Brain Loop with Native Tool Calling
            import json
            print(f"[ADA DEBUG] [BRAIN] Native turn with {self._current_brain_model}...")
            
            # Prepare tools in the format expected by OpenAI/Groq
            # We already have groq_tools prepared in __init__
            
            self.chat_history.append({"role": "user", "content": user_message})
            
            max_tool_turns = 10 # Increase for complex UI tasks
            focus_latched = False
            
            for _turn in range(max_tool_turns):
                chat_completion = await self._create_chat_completion(
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.chat_history,
                    tools=groq_tools,
                    temperature=0.3
                )
                
                response_msg = chat_completion.choices[0].message
                tool_calls = response_msg.tool_calls
                
                # Append assistant message to history (essential for native tool calling)
                # We convert the mock/groq object to a dict for consistency
                history_entry = {"role": "assistant", "content": response_msg.content}
                if tool_calls:
                    history_entry["tool_calls"] = []
                    for tc in tool_calls:
                        history_entry["tool_calls"].append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        })
                self.chat_history.append(history_entry)

                if not tool_calls:
                    # No more tools, this is the final response
                    text_response = response_msg.content or ""
                    break
                
                # Handle Tool Calls
                needs_focus = any(tc.function.name in FOCUS_TOOL_NAMES for tc in tool_calls)
                if needs_focus and not focus_latched:
                    await self._set_focus_mode(True)
                    focus_latched = True

                for tc in tool_calls:
                    tool_name = tc.function.name
                    raw_args = tc.function.arguments
                    
                    # Parse args (might be string or dict depending on provider)
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except:
                            args = {}
                    else:
                        args = raw_args

                    print(f"[ADA DEBUG] [TOOL] Native Execution: {tool_name}({args})")
                    
                    # Notify UI via transcription as a status
                    if self.on_transcription:
                        status_msg = f"[Action: {tool_name}]"
                        if tool_name == "play_music": status_msg = f"[Playing {args.get('query')}...]"
                        elif tool_name == "mouse_action": status_msg = f"[Clicking screen...]"
                        elif tool_name == "see_desktop": status_msg = f"[Analyzing screen...]"
                        self.on_transcription({"sender": "ADA (Action)", "text": status_msg})
                    
                    # Execute
                    result = await self._execute_pipeline_tool(tool_name, args, manage_focus=False)
                    
                    # Append result to history
                    self.chat_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": str(result)
                    })

            if focus_latched:
                await self._set_focus_mode(False)

            # Final response is now in text_response
            if not text_response:
                text_response = "I have completed the task, Sir."


            # 4. Voice Synthesis (Deepgram Aura-2 / ElevenLabs v3)
            if text_response.strip():
                # Routing Logic & Heuristics
                has_hd_tag = "[VOICE: HD]" in text_response
                has_fast_tag = "[VOICE: FAST]" in text_response
                has_emotional_tags = any(tag in text_response for tag in ["[excited]", "[whispers]", "[laughs]", "[slow]", "[fast]"])
                is_long_response = len(text_response) > 120
                
                # HEURISTIC: Force HD if Brain forgot the tag but used emotions or wrote a lot
                use_hd = has_hd_tag or (not has_fast_tag and (has_emotional_tags or is_long_response))
                
                if use_hd:
                    print(f"\n>>> [ADA VOICE] ROUTING TO ELEVENLABS HD (v3) <<<\n")
                else:
                    print(f"\n>>> [ADA VOICE] ROUTING TO DEEPGRAM FAST (Aura-2) <<<\n")

                clean_display_text = text_response.replace("[VOICE: FAST]", "").replace("[VOICE: HD]", "").strip()
                
                if self.on_transcription:
                    self.on_transcription({"sender": "ADA", "text": clean_display_text})

                output_file = None
                audio_played = False
                
                # Synthesize
                if use_hd:
                    output_file = await self._synthesize_elevenlabs(text_response)
                else:
                    output_file = await self._synthesize_deepgram(text_response)
                
                if output_file:
                    try:
                        if self.on_tts_state: self.on_tts_state(True)
                        pygame.mixer.init()
                        pygame.mixer.music.load(output_file)
                        pygame.mixer.music.play()
                        audio_played = True
                    except Exception as e:
                        print(f"[ADA DEBUG] [ERR] Playback failed: {e}")

                # Unified Cleanup Task
                if audio_played and output_file:
                    async def cleanup_task(path):
                        try:
                            # Wait for playback
                            if pygame.mixer.get_init():
                                while pygame.mixer.music.get_busy(): await asyncio.sleep(0.5)
                            
                            if self.on_tts_state: self.on_tts_state(False)
                            # Unload and remove
                            if hasattr(pygame.mixer.music, 'unload'): pygame.mixer.music.unload()
                            if os.path.exists(path):
                                os.remove(path)
                                print(f"[ADA DEBUG] [CLEANUP] Deleted {path}")
                        except Exception as e:
                            print(f"[ADA DEBUG] [ERR] Cleanup failed for {path}: {e}")
                    
                    asyncio.create_task(cleanup_task(output_file))
            
        except Exception as e:
            print(f"[ADA DEBUG] [ERR] Pipeline Turn Failed: {e}")
            traceback.print_exc()
        finally:
            self.is_processing = False

    async def run(self, start_message=None):
        retry_delay = 1
        is_reconnect = False
        
        while not self.stop_event.is_set():
            try:
                if USE_PIPELINE:
                    print(f"[ADA DEBUG] [CONNECT] Starting ADA in Pipeline Mode ({MODEL_BRAIN} + {MODEL_VOICE})")
                    # No persistent session needed for simple Pipeline
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = None # Not used in pipeline VAD
                    
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(self.listen_audio())
                        if self.video_mode == "camera":
                            tg.create_task(self.get_frames())
                        tg.create_task(self.play_audio())
                        
                        if start_message:
                            # Optional: Generate a starting greeting via TTS
                            pass
                        
                        await self.stop_event.wait()
                else:
                    print(f"[ADA DEBUG] [CONNECT] Connecting to Gemini Live API ({MODEL_LIVE})...")
                    async with (
                        client.aio.live.connect(model=MODEL_LIVE, config=config) as session,
                        asyncio.TaskGroup() as tg,
                    ):
                        self.session = session

                        self.audio_in_queue = asyncio.Queue()
                        self.out_queue = asyncio.Queue(maxsize=10)

                        tg.create_task(self.send_realtime())
                        tg.create_task(self.listen_audio())

                        if self.video_mode == "camera":
                            tg.create_task(self.get_frames())
                        elif self.video_mode == "screen":
                            tg.create_task(self.get_screen())

                        tg.create_task(self.receive_audio())
                        tg.create_task(self.play_audio())

                        # Handle Startup vs Reconnect Logic
                        if not is_reconnect:
                            if start_message:
                                print(f"[ADA DEBUG] [INFO] Sending start message: {start_message}")
                                await self.session.send(input=start_message, end_of_turn=True)
                            
                            # Sync Project State
                            if self.on_project_update and self.project_manager:
                                self.on_project_update(self.project_manager.current_project)
                        
                        else:
                            print(f"[ADA DEBUG] [RECONNECT] Connection restored.")
                            history = self.project_manager.get_recent_chat_history(limit=10)
                            
                            context_msg = "System Notification: Connection was lost and just re-established. Here is the recent chat history to help you resume seamlessly:\n\n"
                            for entry in history:
                                sender = entry.get('sender', 'Unknown')
                                text = entry.get('text', '')
                                context_msg += f"[{sender}]: {text}\n"
                            
                            context_msg += "\nPlease acknowledge the reconnection to the user and resume what you were doing."
                            
                            print(f"[ADA DEBUG] [RECONNECT] Sending restoration context to model...")
                            await self.session.send(input=context_msg, end_of_turn=True)

                        # Reset retry delay on successful connection
                        retry_delay = 1
                        await self.stop_event.wait()

            except asyncio.CancelledError:
                print(f"[ADA DEBUG] [STOP] Main loop cancelled.")
                break
                
            except Exception as e:
                print(f"[ADA DEBUG] [ERR] Connection Error: {e}")
                
                # REPORT QUOTA ERRORS TO FRONTEND
                if "quota" in str(e).lower() or "1011" in str(e):
                    if self.on_error:
                        self.on_error("Gemini Quota Exceeded. Please wait a moment or check your API plan.")
                
                if self.stop_event.is_set():
                    break
                
                print(f"[ADA DEBUG] [RETRY] Reconnecting in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 10)
                is_reconnect = True
                
            finally:
                # Cleanup before retry
                if hasattr(self, 'audio_stream') and self.audio_stream:
                    try:
                        self.audio_stream.close()
                    except: 
                        pass


def get_input_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    devices = []
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            devices.append((i, p.get_device_info_by_host_api_device_index(0, i).get('name')))
    p.terminate()
    return devices

def get_output_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    devices = []
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')) > 0:
            devices.append((i, p.get_device_info_by_host_api_device_index(0, i).get('name')))
    p.terminate()
    return devices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())