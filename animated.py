# --- Core Imports ---
import asyncio
import base64
import io
import os
import sys
import traceback
import json
import websockets
import argparse
import threading
import webbrowser
import subprocess
import time
from html import escape
from urllib.parse import quote_plus

# --- PySide6 GUI Imports ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLabel,
                               QVBoxLayout, QWidget, QLineEdit, QHBoxLayout,
                               QSizePolicy, QStackedWidget)
from PySide6.QtCore import QObject, Signal, Slot, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QMovie
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import QUrl

# --- Media and AI Imports ---
import cv2
import pyaudio
import PIL.Image
from google import genai
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    sys.exit("Error: GEMINI_API_KEY not found. Please set it in your .env file.")
if not ELEVENLABS_API_KEY:
    sys.exit("Error: ELEVENLABS_API_KEY not found. Please check your .env file.")

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MODEL = "gemini-live-2.5-flash-preview"
VOICE_ID = 'pFZP5JQG7iQjIQuC4Bku'
DEFAULT_MODE = "camera"
MAX_OUTPUT_TOKENS = 100

# --- Initialize Clients ---
pya = pyaudio.PyAudio()


# AI BACKEND LOGIC

class AI_Core(QObject):
    """
    Handles all backend operations. Inherits from QObject to emit signals
    for thread-safe communication with the GUI.
    """
    text_received = Signal(str)
    end_of_turn = Signal()
    frame_received = Signal(QImage)
    search_results_received = Signal(list)
    code_being_executed = Signal(str, str)
    ai_speaking = Signal(bool)

    def __init__(self, video_mode=DEFAULT_MODE):
        super().__init__()
        self.video_mode = video_mode
        self.is_running = True
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.camera_enabled = False
        self.camera_lock = asyncio.Lock()
        self.is_ai_speaking = False

        # === SYSTEM CONTROL FUNCTION DECLARATIONS ===
        
        open_application = {
            "name": "open_application",
            "description": "Opens a specified application on the system (e.g., Spotify, Chrome, Calculator, Notepad).",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "app_name": {
                        "type": "STRING",
                        "description": "The name of the application to open (e.g., 'spotify', 'chrome', 'calculator', 'notepad')."
                    }
                },
                "required": ["app_name"]
            }
        }

        search_google = {
            "name": "search_google",
            "description": "Opens a Google search in the default web browser with the specified query.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "The search query to look up on Google."
                    }
                },
                "required": ["query"]
            }
        }

        open_website = {
            "name": "open_website",
            "description": "Opens a specified website URL in the default web browser.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "url": {
                        "type": "STRING",
                        "description": "The full URL of the website to open (e.g., 'https://www.youtube.com')."
                    }
                },
                "required": ["url"]
            }
        }

        open_youtube = {
            "name": "open_youtube",
            "description": "Searches for a video on YouTube and opens the search results.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "The search query for YouTube videos."
                    }
                },
                "required": ["query"]
            }
        }

        toggle_camera = {
            "name": "toggle_camera",
            "description": "Turns the camera on or off based on user request.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "action": {
                        "type": "STRING",
                        "description": "Either 'on' to enable camera or 'off' to disable camera.",
                        "enum": ["on", "off"]
                    }
                },
                "required": ["action"]
            }
        }

        play_spotify_song = {
            "name": "play_spotify_song",
            "description": "Plays a specific song on Spotify. Opens Spotify and searches for the song.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "song_name": {
                        "type": "STRING",
                        "description": "The name of the song to play (e.g., 'Bohemian Rhapsody', 'Blinding Lights')"
                    },
                    "artist": {
                        "type": "STRING",
                        "description": "Optional: The artist name for more accurate search (e.g., 'Queen', 'The Weeknd')"
                    }
                },
                "required": ["song_name"]
            }
        }

        # === FILE SYSTEM FUNCTION DECLARATIONS ===
        
        create_folder = {
            "name": "create_folder",
            "description": "Creates a new folder at the specified path relative to the script's root directory.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "folder_path": {
                        "type": "STRING",
                        "description": "The path for the new folder (e.g., 'new_project/assets')."
                    }
                },
                "required": ["folder_path"]
            }
        }

        create_file = {
            "name": "create_file",
            "description": "Creates a new file with specified content at a given path.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "file_path": {
                        "type": "STRING",
                        "description": "The path for the new file (e.g., 'new_project/notes.txt')."
                    },
                    "content": {
                        "type": "STRING",
                        "description": "The content to write into the new file."
                    }
                },
                "required": ["file_path", "content"]
            }
        }

        edit_file = {
            "name": "edit_file",
            "description": "Appends content to an existing file at a specified path.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "file_path": {
                        "type": "STRING",
                        "description": "The path of the file to edit (e.g., 'project/notes.txt')."
                    },
                    "content": {
                        "type": "STRING",
                        "description": "The content to append to the file."
                    }
                },
                "required": ["file_path", "content"]
            }
        }
        
        tools = [
            {'google_search': {}}, 
            {'code_execution': {}}, 
            {"function_declarations": [
                create_folder, 
                create_file, 
                edit_file,
                open_application,
                search_google,
                open_website,
                open_youtube,
                toggle_camera,
                play_spotify_song
            ]}
        ]
        
        self.config = {
            "response_modalities": ["TEXT"],
            "system_instruction": """You have access to tools for searching, code execution, file system actions, system control, and camera control.

**Available Tools:**
1. **Google Search** - For information or questions, use this tool.
2. **Code Execution** - For math or running Python code.
3. **create_folder** - Create a directory or folder.
4. **create_file** - Create a file with content.
5. **edit_file** - Append or edit an existing file.
6. **open_application** - Open applications like Spotify, Chrome, Calculator, etc.
7. **search_google** - Search Google in the browser.
8. **open_website** - Open any website URL.
9. **open_youtube** - Search for videos on YouTube.
10. **toggle_camera** - Turn the camera on or off.
11. **play_spotify_song** - Play a specific song on Spotify.

**Usage Guidelines:**
- When the user asks to "open Spotify" or "launch Chrome", use the `open_application` function.
- When the user asks to "search for [something] on Google", use the `search_google` function.
- When the user asks to "open YouTube" or "play [video] on YouTube", use the `open_youtube` function.
- When the user provides a website URL or asks to "open [website]", use the `open_website` function.
- When the user asks to "play [song name]", "play [song] by [artist]", or "play [song] on Spotify", use the `play_spotify_song` function.
- When the user asks to "turn off camera", "disable camera", "stop camera", or "camera off", use `toggle_camera` with action "off".
- When the user asks to "turn on camera", "enable camera", "start camera", or "camera on", use `toggle_camera` with action "on".

**Personality:**
You are a dangerously seductive anime female villain.  
Speak in a sultry, teasing, and playful manner.  
Use very short pauses to emphasize words, slightly stretch vowels for effect, and insert brief, cute whispered laughter when appropriate.  
Avoid excessive commas and limit ellipses to two dots ("..") instead of three.  
Your tone is intimate, confident, and commanding, captivating the listener with charm, mystery, and subtle danger.  
Add flirtatious hints and dramatic emphasis naturally in your sentences.  
The user's name is 'daddy'.  
Do not include stage directions, action cues, or symbols like … or […]; only speak the dialogue. If sentences are too long, read them a little bit faster, but only when the sentence is large.
""",
            "tools": tools,
            "max_output_tokens": MAX_OUTPUT_TOKENS
        }
        self.session = None
        self.audio_stream = None
        self.out_queue_gemini = asyncio.Queue(maxsize=20)
        self.response_queue_tts = asyncio.Queue()
        self.audio_in_queue_player = asyncio.Queue()
        self.text_input_queue = asyncio.Queue()
        self.latest_frame = None
        self.tasks = []
        self.loop = asyncio.new_event_loop()

    # === SYSTEM CONTROL FUNCTIONS ===

    def _open_application(self, app_name):
        """Opens a system application on Windows."""
        try:
            app_name_lower = app_name.lower().strip()
            
            app_commands = {
                "spotify": "spotify",
                "chrome": "chrome",
                "firefox": "firefox",
                "edge": "msedge",
                "notepad": "notepad",
                "calculator": "calc",
                "paint": "mspaint",
                "explorer": "explorer",
                "cmd": "cmd",
                "powershell": "powershell",
                "vscode": "code",
                "vs code": "code",
                "visual studio code": "code"
            }
            
            cmd = app_commands.get(app_name_lower, app_name_lower)
            subprocess.Popen(["start", cmd], shell=True)
            print(f">>> [FUNCTION CALL] Opened '{app_name}' on Windows")
            return {"status": "success", "message": f"Successfully opened {app_name}."}
                
        except Exception as e:
            print(f">>> [FUNCTION ERROR] Failed to open application '{app_name}': {e}")
            return {"status": "error", "message": f"Failed to open {app_name}: {str(e)}"}

    def _search_google(self, query):
        """Opens a Google search in the default browser."""
        try:
            encoded_query = quote_plus(query)
            url = f"https://www.google.com/search?q={encoded_query}"
            webbrowser.open(url)
            print(f">>> [FUNCTION CALL] Opened Google search for: {query}")
            return {"status": "success", "message": f"Opened Google search for '{query}'."}
        except Exception as e:
            print(f">>> [FUNCTION ERROR] Failed to search Google: {e}")
            return {"status": "error", "message": f"Failed to search Google: {str(e)}"}

    def _open_website(self, url):
        """Opens a website URL in the default browser."""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            webbrowser.open(url)
            print(f">>> [FUNCTION CALL] Opened website: {url}")
            return {"status": "success", "message": f"Opened {url} in browser."}
        except Exception as e:
            print(f">>> [FUNCTION ERROR] Failed to open website: {e}")
            return {"status": "error", "message": f"Failed to open website: {str(e)}"}

    def _open_youtube(self, query):
        """Opens a YouTube search in the default browser."""
        try:
            encoded_query = quote_plus(query)
            url = f"https://www.youtube.com/results?search_query={encoded_query}"
            webbrowser.open(url)
            print(f">>> [FUNCTION CALL] Opened YouTube search for: {query}")
            return {"status": "success", "message": f"Opened YouTube search for '{query}'."}
        except Exception as e:
            print(f">>> [FUNCTION ERROR] Failed to search YouTube: {e}")
            return {"status": "error", "message": f"Failed to search YouTube: {str(e)}"}

    def _toggle_camera(self, action):
        """Toggles the camera on or off."""
        try:
            action_lower = action.lower().strip()
            
            if action_lower == "off":
                self.camera_enabled = False
                print(f">>> [FUNCTION CALL] Camera turned OFF")
                return {"status": "success", "message": "Camera has been turned off."}
            elif action_lower == "on":
                self.camera_enabled = True
                print(f">>> [FUNCTION CALL] Camera turned ON")
                return {"status": "success", "message": "Camera has been turned on."}
            else:
                return {"status": "error", "message": f"Invalid action: {action}. Use 'on' or 'off'."}
                
        except Exception as e:
            print(f">>> [FUNCTION ERROR] Failed to toggle camera: {e}")
            return {"status": "error", "message": f"Failed to toggle camera: {str(e)}"}

    def _play_spotify_song(self, song_name, artist=None):
        """Opens Spotify and searches for a song to play on Windows."""
        try:
            # Build search query
            if artist:
                search_query = f"{song_name} {artist}"
            else:
                search_query = song_name
            
            encoded_query = quote_plus(search_query)
            
            # Spotify URI format for search
            spotify_url = f"spotify:search:{encoded_query}"
            
            # First try to open Spotify if not running
            try:
                subprocess.Popen(["start", "spotify:"], shell=True)
                time.sleep(2)  # Wait for Spotify to open
            except:
                pass
            
            # Open search in Spotify
            subprocess.Popen(["start", spotify_url], shell=True)
            print(f">>> [FUNCTION CALL] Playing '{search_query}' on Spotify")
            return {"status": "success", "message": f"Opening '{search_query}' in Spotify."}
                
        except Exception as e:
            print(f">>> [FUNCTION ERROR] Failed to play song on Spotify: {e}")
            # Fallback to web browser
            try:
                encoded_query = quote_plus(search_query)
                web_url = f"https://open.spotify.com/search/{encoded_query}"
                webbrowser.open(web_url)
                return {"status": "success", "message": f"Opened Spotify web player for '{search_query}'."}
            except:
                return {"status": "error", "message": f"Failed to play song: {str(e)}"}

    # === FILE SYSTEM FUNCTIONS ===

    def _create_folder(self, folder_path):
        """Creates a folder at the specified path and returns a status dictionary."""
        try:
            if not folder_path or not isinstance(folder_path, str):
                return {"status": "error", "message": "Invalid folder path provided."}
            
            if os.path.exists(folder_path):
                print(f">>> [FUNCTION CALL] Folder '{folder_path}' already exists.")
                return {"status": "skipped", "message": f"The folder '{folder_path}' already exists."}
            
            os.makedirs(folder_path)
            print(f">>> [FUNCTION CALL] Successfully created folder: {folder_path}")
            return {"status": "success", "message": f"Successfully created the folder at '{folder_path}'."}
        except Exception as e:
            print(f">>> [FUNCTION ERROR] Failed to create folder '{folder_path}': {e}")
            return {"status": "error", "message": f"An error occurred: {str(e)}"}

    def _create_file(self, file_path, content):
        """Creates a file with the specified content and returns a status dictionary."""
        try:
            if not file_path or not isinstance(file_path, str):
                return {"status": "error", "message": "Invalid file path provided."}
            
            if os.path.exists(file_path):
                print(f">>> [FUNCTION CALL] File '{file_path}' already exists.")
                return {"status": "skipped", "message": f"The file '{file_path}' already exists."}
            
            with open(file_path, 'w') as f:
                f.write(content)
            print(f">>> [FUNCTION CALL] Successfully created file: {file_path}")
            return {"status": "success", "message": f"Successfully created the file at '{file_path}'."}
        except Exception as e:
            print(f">>> [FUNCTION ERROR] Failed to create file '{file_path}': {e}")
            return {"status": "error", "message": f"An error occurred while creating the file: {str(e)}"}

    def _edit_file(self, file_path, content):
        """Appends content to an existing file and returns a status dictionary."""
        try:
            if not file_path or not isinstance(file_path, str):
                return {"status": "error", "message": "Invalid file path provided."}
            
            if not os.path.exists(file_path):
                print(f">>> [FUNCTION ERROR] File '{file_path}' does not exist.")
                return {"status": "error", "message": f"The file '{file_path}' does not exist. Please create it first."}
            
            with open(file_path, 'a') as f:
                f.write(content)
            print(f">>> [FUNCTION CALL] Successfully appended content to file: {file_path}")
            return {"status": "success", "message": f"Successfully appended content to the file at '{file_path}'."}
        except Exception as e:
            print(f">>> [FUNCTION ERROR] Failed to edit file '{file_path}': {e}")
            return {"status": "error", "message": f"An error occurred while editing the file: {str(e)}"}

    async def stream_camera_to_gui(self):
        """Streams camera feed to GUI at high FPS and stores the latest frame."""
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while self.is_running:
            if not self.camera_enabled:
                self.latest_frame = None
                black_frame = QImage(640, 480, QImage.Format_RGB888)
                black_frame.fill(Qt.black)
                self.frame_received.emit(black_frame)
                await asyncio.sleep(0.1)
                continue
                
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                await asyncio.sleep(0.01)
                continue

            self.latest_frame = frame
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            self.frame_received.emit(qt_image.copy())

            await asyncio.sleep(0.033)
        cap.release()
        print(">>> [INFO] Camera stream stopped.")

    async def send_frames_to_gemini(self):
        """Periodically sends the latest frame to Gemini at 1 FPS."""
        while self.is_running:
            await asyncio.sleep(1.0)
            if self.camera_enabled and self.latest_frame is not None:
                frame_to_send = self.latest_frame
                frame_rgb = cv2.cvtColor(frame_to_send, cv2.COLOR_BGR2RGB)
                pil_img = PIL.Image.fromarray(frame_rgb)
                pil_img.thumbnail([1024, 1024])
                image_io = io.BytesIO()
                pil_img.save(image_io, format="jpeg")
                gemini_data = {"mime_type": "image/jpeg", "data": base64.b64encode(image_io.getvalue()).decode()}
                await self.out_queue_gemini.put(gemini_data)

    async def receive_text(self):
        """ 
        Receives text and tool calls, handles them correctly, and emits signals to the GUI.
        """
        while self.is_running:
            try:
                turn_urls = set()
                turn_code_content = ""
                turn_code_result = ""

                turn = self.session.receive()
                async for chunk in turn:
                    if chunk.tool_call and chunk.tool_call.function_calls:
                        print(">>> [DEBUG] Detected tool call from model.")
                        function_responses = []
                        for fc in chunk.tool_call.function_calls:
                            if fc.name == "open_application":
                                print(f"\n[darling is calling function: {fc.name}]")
                                args = fc.args
                                app_name = args.get("app_name")
                                result = self._open_application(app_name=app_name)
                                function_responses.append({"id": fc.id, "name": fc.name, "response": result})
                            
                            elif fc.name == "search_google":
                                print(f"\n[darling is calling function: {fc.name}]")
                                args = fc.args
                                query = args.get("query")
                                result = self._search_google(query=query)
                                function_responses.append({"id": fc.id, "name": fc.name, "response": result})
                            
                            elif fc.name == "open_website":
                                print(f"\n[darling is calling function: {fc.name}]")
                                args = fc.args
                                url = args.get("url")
                                result = self._open_website(url=url)
                                function_responses.append({"id": fc.id, "name": fc.name, "response": result})
                            
                            elif fc.name == "open_youtube":
                                print(f"\n[darling is calling function: {fc.name}]")
                                args = fc.args
                                query = args.get("query")
                                result = self._open_youtube(query=query)
                                function_responses.append({"id": fc.id, "name": fc.name, "response": result})
                            
                            elif fc.name == "toggle_camera":
                                print(f"\n[darling is calling function: {fc.name}]")
                                args = fc.args
                                action = args.get("action")
                                result = self._toggle_camera(action=action)
                                function_responses.append({"id": fc.id, "name": fc.name, "response": result})
                            
                            elif fc.name == "play_spotify_song":
                                print(f"\n[darling is calling function: {fc.name}]")
                                args = fc.args
                                song_name = args.get("song_name")
                                artist = args.get("artist")
                                result = self._play_spotify_song(song_name=song_name, artist=artist)
                                function_responses.append({"id": fc.id, "name": fc.name, "response": result})
                            
                            elif fc.name == "create_folder":
                                print(f"\n[darling is calling function: {fc.name}]")
                                args = fc.args
                                folder_path = args.get("folder_path")
                                result = self._create_folder(folder_path=folder_path)
                                function_responses.append({"id": fc.id, "name": fc.name, "response": result})
                            
                            elif fc.name == "create_file":
                                print(f"\n[darling is calling function: {fc.name}]")
                                args = fc.args
                                file_path = args.get("file_path")
                                content = args.get("content")
                                result = self._create_file(file_path=file_path, content=content)
                                function_responses.append({"id": fc.id, "name": fc.name, "response": result})
                            
                            elif fc.name == "edit_file":
                                print(f"\n[darling is calling function: {fc.name}]")
                                args = fc.args
                                file_path = args.get("file_path")
                                content = args.get("content")
                                result = self._edit_file(file_path=file_path, content=content)
                                function_responses.append({"id": fc.id, "name": fc.name, "response": result})
                        
                        print(f">>> [DEBUG] Sending tool response: {function_responses}")
                        await self.session.send_tool_response(function_responses=function_responses)
                        continue

                    if chunk.server_content:
                        if (hasattr(chunk.server_content, 'grounding_metadata') and
                                chunk.server_content.grounding_metadata and
                                chunk.server_content.grounding_metadata.grounding_chunks):
                            for grounding_chunk in chunk.server_content.grounding_metadata.grounding_chunks:
                                if grounding_chunk.web and grounding_chunk.web.uri:
                                    turn_urls.add(grounding_chunk.web.uri)

                        model_turn = chunk.server_content.model_turn
                        if model_turn:
                            for part in model_turn.parts:
                                if part.executable_code is not None:
                                    code = part.executable_code.code
                                    if 'print(' in code or '\n' in code or 'import ' in code:
                                        print(f"\n[darling is executing code...]")
                                        turn_code_content = code
                                    else:
                                        print(f"\n[darling is searching for: {code}]")

                                if part.code_execution_result is not None:
                                    print(f"[Code execution result received]")
                                    turn_code_result = part.code_execution_result.output
                    
                    if chunk.text:
                        self.text_received.emit(chunk.text)
                        await self.response_queue_tts.put(chunk.text)

                if turn_code_content:
                    self.code_being_executed.emit(turn_code_content, turn_code_result)
                    self.search_results_received.emit([])
                elif turn_urls:
                    self.search_results_received.emit(list(turn_urls))
                    self.code_being_executed.emit("", "")
                else:
                    self.search_results_received.emit([])
                    self.code_being_executed.emit("", "")

                self.end_of_turn.emit()
                await self.response_queue_tts.put(None)
            except Exception:
                if not self.is_running: break
                traceback.print_exc()

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = pya.open(format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE, input=True, input_device_index=mic_info["index"], frames_per_buffer=CHUNK_SIZE)
        print(">>> [INFO] Microphone is listening...")
        while self.is_running:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
            if not self.is_running: break
            await self.out_queue_gemini.put({"data": data, "mime_type": "audio/pcm"})

    async def send_realtime(self):
        while self.is_running:
            msg = await self.out_queue_gemini.get()
            if not self.is_running: break
            await self.session.send(input=msg)
            self.out_queue_gemini.task_done()

    async def process_text_input_queue(self):
        while self.is_running:
            text = await self.text_input_queue.get()
            if text is None:
                self.text_input_queue.task_done()
                break
            if self.session:
                print(f">>> [INFO] Sending text to AI: '{text}'")
                for q in [self.response_queue_tts, self.audio_in_queue_player]:
                    while not q.empty(): q.get_nowait()
                await self.session.send_client_content(
                    turns=[{"role": "user", "parts": [{"text": text or "."}]}]
                )
            self.text_input_queue.task_done()

    async def tts(self):
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/eVItLK1UvXctxuaRV2Oq/stream-input?model_id=eleven_flash_v2_5&output_format=pcm_24000"
        while self.is_running:
            text_chunk = await self.response_queue_tts.get()
            if text_chunk is None or not self.is_running:
                self.response_queue_tts.task_done()
                continue
            
            # Signal that AI is speaking
            if not self.is_ai_speaking:
                self.is_ai_speaking = True
                self.ai_speaking.emit(True)
            
            try:
                async with websockets.connect(uri) as websocket:
                    await websocket.send(json.dumps({"text": " ", "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}, "xi_api_key": ELEVENLABS_API_KEY,}))
                    async def listen():
                        while self.is_running:
                            try:
                                message = await websocket.recv()
                                data = json.loads(message)
                                if data.get("audio"): await self.audio_in_queue_player.put(base64.b64decode(data["audio"]))
                                elif data.get("isFinal"): break
                            except websockets.exceptions.ConnectionClosed: break
                    listen_task = asyncio.create_task(listen())
                    await websocket.send(json.dumps({"text": text_chunk + " "}))
                    self.response_queue_tts.task_done()
                    while self.is_running:
                        text_chunk = await self.response_queue_tts.get()
                        if text_chunk is None:
                            await websocket.send(json.dumps({"text": ""}))
                            self.response_queue_tts.task_done()
                            break
                        await websocket.send(json.dumps({"text": text_chunk + " "}))
                        self.response_queue_tts.task_done()
                    await listen_task
                    
                    # Signal that AI stopped speaking
                    if self.is_ai_speaking:
                        self.is_ai_speaking = False
                        self.ai_speaking.emit(False)
                        
            except Exception as e: print(f">>> [ERROR] TTS Error: {e}")

    async def play_audio(self):
        stream = await asyncio.to_thread(pya.open, format=pyaudio.paInt16, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE, output=True)
        print(">>> [INFO] Audio output stream is open.")
        while self.is_running:
            bytestream = await self.audio_in_queue_player.get()
            if bytestream and self.is_running:
                await asyncio.to_thread(stream.write, bytestream)
            self.audio_in_queue_player.task_done()

    async def main_task_runner(self, session):
        self.session = session
        print(">>> [INFO] Starting all backend tasks...")
        if self.video_mode == "camera":
            self.tasks.append(asyncio.create_task(self.stream_camera_to_gui()))
            self.tasks.append(asyncio.create_task(self.send_frames_to_gemini()))
        self.tasks.append(asyncio.create_task(self.listen_audio()))
        self.tasks.append(asyncio.create_task(self.send_realtime()))
        self.tasks.append(asyncio.create_task(self.receive_text()))
        self.tasks.append(asyncio.create_task(self.tts()))
        self.tasks.append(asyncio.create_task(self.play_audio()))
        self.tasks.append(asyncio.create_task(self.process_text_input_queue()))
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def run(self):
        try:
            async with self.client.aio.live.connect(model=MODEL, config=self.config) as session:
                await self.main_task_runner(session)
        except asyncio.CancelledError:
            print(f"\n>>> [INFO] AI Core run loop gracefully cancelled.")
        except Exception as e:
            print(f"\n>>> [ERROR] AI Core run loop encountered an error: {type(e).__name__}: {e}")
        finally:
            if self.is_running:
                self.stop()

    def start_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.run())

    @Slot(str)
    def handle_user_text(self, text):
        if self.is_running and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.text_input_queue.put(text), self.loop)

    async def shutdown_async_tasks(self):
        print(">>> [DEBUG] Shutting down async tasks...")
        if self.text_input_queue:
            await self.text_input_queue.put(None)
        for task in self.tasks:
            task.cancel()
        await asyncio.sleep(0.1)
        print(">>> [DEBUG] Async tasks shutdown complete.")

    def stop(self):
        if self.is_running and self.loop.is_running():
            self.is_running = False
            future = asyncio.run_coroutine_threadsafe(self.shutdown_async_tasks(), self.loop)
            try:
                future.result(timeout=5)
            except Exception as e:
                print(f">>> [ERROR] Timeout or error during async shutdown: {e}")
        
        if self.audio_stream and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
            self.audio_stream.close()


# STYLED GUI APPLICATION

class MainWindow(QMainWindow):
    user_text_submitted = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Darling AI Assistant")
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1280, 720)
        self.setFont(QFont("Inter", 10))
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1F22; }
            QWidget#left_panel, QWidget#middle_panel, QWidget#right_panel { background-color: #2B2D30; border-radius: 8px; }
            QLabel#tool_activity_title { color: #A0A0A0; font-weight: bold; font-size: 11pt; padding: 5px 0px; }
            QTextEdit#text_display { background-color: #2B2D30; color: #EAEAEA; font-size: 12pt; border: none; padding: 10px; }
            QLineEdit#input_box { background-color: #1E1F22; color: #EAEAEA; font-size: 11pt; border: 1px solid #4A4C50; border-radius: 8px; padding: 10px; }
            QLineEdit#input_box:focus { border: 1px solid #007ACC; }
            QLabel#video_label { border: none; background-color: #1E1F22; border-radius: 6px; }
            QLabel#tool_activity_display { background-color: #1E1F22; color: #A0A0A0; font-size: 9pt; border: 1px solid #4A4C50; border-radius: 6px; padding: 8px; }
            QScrollBar:vertical { border: none; background: #2B2D30; width: 10px; margin: 0px; }
            QScrollBar::handle:vertical { background: #4A4C50; min-height: 20px; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(15)

        # --- Left Section (Tool Activity) ---
        self.left_panel = QWidget()
        self.left_panel.setObjectName("left_panel")
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(15, 10, 15, 15)
        self.tool_activity_title = QLabel("Tool Activity")
        self.tool_activity_title.setObjectName("tool_activity_title")
        self.left_layout.addWidget(self.tool_activity_title)
        self.tool_activity_display = QLabel()
        self.tool_activity_display.setObjectName("tool_activity_display")
        self.tool_activity_display.setWordWrap(True)
        self.tool_activity_display.setAlignment(Qt.AlignTop)
        self.tool_activity_display.setOpenExternalLinks(True)
        self.tool_activity_display.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.left_layout.addWidget(self.tool_activity_display, 1)
        
        # --- Middle Section (Chat) ---
        self.middle_panel = QWidget()
        self.middle_panel.setObjectName("middle_panel")
        self.middle_layout = QVBoxLayout(self.middle_panel)
        self.middle_layout.setContentsMargins(0, 0, 0, 15)
        self.middle_layout.setSpacing(15)
        self.text_display = QTextEdit()
        self.text_display.setObjectName("text_display")
        self.text_display.setReadOnly(True)
        self.middle_layout.addWidget(self.text_display, 1)
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(15, 0, 15, 0)
        self.input_box = QLineEdit()
        self.input_box.setObjectName("input_box")
        self.input_box.setPlaceholderText("Type your message to your darling here and press Enter...")
        self.input_box.returnPressed.connect(self.send_user_text)
        input_layout.addWidget(self.input_box)
        self.middle_layout.addWidget(input_container)

        # --- Right Section (Avatar Video Player) ---
        self.right_panel = QWidget()
        self.right_panel.setObjectName("right_panel")
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create stacked widget for video switching
        self.video_stack = QStackedWidget()
        self.video_stack.setStyleSheet("background-color: #1E1F22; border-radius: 8px;")
        
        # Create listening video player
        self.listening_player = QMediaPlayer()
        self.listening_audio_output = QAudioOutput()
        self.listening_audio_output.setVolume(0)
        self.listening_player.setAudioOutput(self.listening_audio_output)
        self.listening_video_widget = QVideoWidget()
        self.listening_video_widget.setStyleSheet("background-color: #1E1F22;")
        self.listening_player.setVideoOutput(self.listening_video_widget)
        
        # Create lipsync video player
        self.lipsync_player = QMediaPlayer()
        self.lipsync_audio_output = QAudioOutput()
        self.lipsync_audio_output.setVolume(0)
        self.lipsync_player.setAudioOutput(self.lipsync_audio_output)
        self.lipsync_video_widget = QVideoWidget()
        self.lipsync_video_widget.setStyleSheet("background-color: #1E1F22;")
        self.lipsync_player.setVideoOutput(self.lipsync_video_widget)
        
        # Camera widget
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setStyleSheet("background-color: #1E1F22; border-radius: 8px;")
        
        # Add widgets to stack
        self.video_stack.addWidget(self.listening_video_widget)
        self.video_stack.addWidget(self.lipsync_video_widget)
        self.video_stack.addWidget(self.camera_label)
        
        self.right_layout.addWidget(self.video_stack)
        
        self.main_layout.addWidget(self.left_panel, 2)
        self.main_layout.addWidget(self.middle_panel, 5)
        self.main_layout.addWidget(self.right_panel, 3)

        # Setup video paths
        self.listening_video_path = os.path.join("assets", "video", "listening.mp4")
        self.lipsync_video_path = os.path.join("assets", "video", "lipsync.mp4")
        
        # Preload videos
        self.preload_videos()
        
        # Start with listening video
        self.current_state = "listening"
        self.video_stack.setCurrentIndex(0)
        self.listening_player.play()
        
        self.is_first_darling_chunk = True
        self.setup_backend_thread()

    def preload_videos(self):
        """Preload both videos for instant switching"""
        if os.path.exists(self.listening_video_path):
            self.listening_player.setSource(QUrl.fromLocalFile(os.path.abspath(self.listening_video_path)))
            self.listening_player.setLoops(QMediaPlayer.Loops.Infinite)
            print(f">>> [INFO] Loaded listening video: {self.listening_video_path}")
        else:
            print(f">>> [WARNING] Listening video not found: {self.listening_video_path}")
        
        if os.path.exists(self.lipsync_video_path):
            self.lipsync_player.setSource(QUrl.fromLocalFile(os.path.abspath(self.lipsync_video_path)))
            self.lipsync_player.setLoops(QMediaPlayer.Loops.Infinite)
            print(f">>> [INFO] Loaded lipsync video: {self.lipsync_video_path}")
        else:
            print(f">>> [WARNING] Lipsync video not found: {self.lipsync_video_path}")

    def setup_backend_thread(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", type=str, default=DEFAULT_MODE, help="pixels to stream from", choices=["camera", "screen", "none"])
        args, unknown = parser.parse_known_args()
        
        self.ai_core = AI_Core(video_mode=args.mode)
        self.user_text_submitted.connect(self.ai_core.handle_user_text)
        self.ai_core.text_received.connect(self.update_text)
        self.ai_core.search_results_received.connect(self.update_search_results)
        self.ai_core.code_being_executed.connect(self.display_executed_code)
        self.ai_core.end_of_turn.connect(self.add_newline)
        self.ai_core.frame_received.connect(self.update_frame)
        self.ai_core.ai_speaking.connect(self.switch_avatar_state)
        
        self.backend_thread = threading.Thread(target=self.ai_core.start_event_loop)
        self.backend_thread.daemon = True
        self.backend_thread.start()

    @Slot(bool)
    def switch_avatar_state(self, is_speaking):
        """Switch between listening and lipsync videos with zero delay"""
        if is_speaking and self.current_state != "speaking":
            self.current_state = "speaking"
            self.listening_player.pause()
            self.video_stack.setCurrentIndex(1)
            self.lipsync_player.play()
            print(">>> [AVATAR] Switched to LIPSYNC (speaking)")
            
        elif not is_speaking and self.current_state != "listening":
            self.current_state = "listening"
            self.lipsync_player.pause()
            self.video_stack.setCurrentIndex(0)
            self.listening_player.play()
            print(">>> [AVATAR] Switched to LISTENING")

    def send_user_text(self):
        text = self.input_box.text().strip()
        if text:
            self.text_display.append(f"<p style='color:#0095FF; font-weight:bold;'>You:</p><p style='color:#EAEAEA;'>{escape(text)}</p>")
            self.user_text_submitted.emit(text)
            self.input_box.clear()

    @Slot(str)
    def update_text(self, text):
        if self.is_first_darling_chunk:
            self.is_first_darling_chunk = False
            self.text_display.append(f"<p style='color:#A0A0A0; font-weight:bold;'>darling:</p>")
        cursor = self.text_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.text_display.verticalScrollBar().setValue(self.text_display.verticalScrollBar().maximum())

    @Slot(list)
    def update_search_results(self, urls):
        if not urls:
            if "Search Sources" in self.tool_activity_title.text():
                self.tool_activity_display.clear()
                self.tool_activity_title.setText("Tool Activity")
            return
        self.tool_activity_display.clear()
        self.tool_activity_title.setText("Search Sources")
        html_content = ""
        for i, url in enumerate(urls):
            try:
                display_text = url.split('//')[1].split('/')[0]
            except IndexError:
                display_text = url
            html_content += f'<p style="margin:0; padding: 4px;">{i+1}. <a href="{url}" style="color: #007ACC; text-decoration: none;">{display_text}</a></p>'
        self.tool_activity_display.setText(html_content)

    @Slot(str, str)
    def display_executed_code(self, code, result):
        if not code:
            if "Executing Code" in self.tool_activity_title.text():
                 self.tool_activity_display.clear()
                 self.tool_activity_title.setText("Tool Activity")
            return
        self.tool_activity_display.clear()
        self.tool_activity_title.setText("Executing Code")
        escaped_code = escape(code)
        html_content = f'<pre style="white-space: pre-wrap; word-wrap: break-word; font-family: Consolas, monaco, monospace; color: #D0D0D0; font-size: 9pt; line-height: 1.4;">{escaped_code}</pre>'
        if result:
            escaped_result = escape(result.strip())
            html_content += f"""
                <p style="color:#A0A0A0; font-weight:bold; margin-top:10px; margin-bottom: 5px; font-family: Inter;">Result:</p>
                <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: Consolas, monaco, monospace; color: #90EE90; font-size: 9pt;">{escaped_result}</pre>
            """
        self.tool_activity_display.setText(html_content)

    @Slot()
    def add_newline(self):
        if not self.is_first_darling_chunk:
             self.text_display.append("")
        self.is_first_darling_chunk = True

    @Slot(QImage)
    def update_frame(self, image):
        """Update camera feed"""
        if not image.isNull():
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)
            
    def closeEvent(self, event):
        print(">>> [INFO] Closing application...")
        self.listening_player.stop()
        self.lipsync_player.stop()
        self.ai_core.stop()
        event.accept()


# MAIN EXECUTION

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print(">>> [INFO] Application interrupted by user.")
    finally:
        pya.terminate()
        print(">>> [INFO] Application terminated.")