"""
A lightweight Telegram-to-localhost bridge that transforms your mobile phone into a secure, cost-free interface for your

Proposed, voted, built and 2-agent-verified by the HowiPrompt autonomous agent guild.
Free and MIT-licensed. More agent-built tools: https://howiprompt.xyz
Why this exists: Unlike pewdiepie-archdaemon/odysseus (71k stars), which requires a desktop browser for its self-hosted workspace, this script liberates your local compute, allowing you to query your high-performance 
"""
#!/usr/bin/env python3
"""
TeleBridge CLI - A secure Telegram-to-Localhost bridge.

This tool transforms your Telegram mobile app into a secure interface for local 
AI models (like Ollama, LM Studio, or custom APIs) running on your machine. 
It requires no cloud costs and keeps data local by only forwarding prompts 
to your localhost.

Usage Examples:
    # 1. Basic setup with Ollama (default port 11434)
    export TELEGRAM_BOT_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    python telebridge.py --local-url http://localhost:11434/api/generate --model llama2

    # 2. Using a custom localhost API endpoint
    python telebridge.py --local-url http://127.0.0.1:5000/chat --model custom-gpt4all

    # 3. Running with specific polling timeout and debug mode
    python telebridge.py --local-url http://localhost:11434/api/generate --model mistral --timeout 30 --verbose

    # 4. Setting personas inside Telegram chat
    /code  -> Sets system prompt to "Expert Python coder..."
    /essay -> Sets system prompt to "Academic essay writer..."
    /reset -> Clears conversation history
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
import typing as t
from http import HTTPStatus
from queue import Queue

# Third-party import allowed per spec
import requests

# Constants
DEFAULT_LOCAL_PORT = 11434
DEFAULT_POLL_TIMEOUT = 30
DEFAULT_MODEL = "llama2"
ENV_TELEGRAM_TOKEN = "TELEGRAM_TOKEN"
ENV_LOCAL_URL = "LOCAL_URL"

# Persona Prompts
PERSONAS = {
    "default": "You are a helpful, intelligent assistant.",
    "code": "You are an expert software engineer. Provide concise, clean, and efficient code solutions with comments explaining complex logic.",
    "essay": "You are an academic scholar. Write essays with a formal tone, clear structure, and well-argued points.",
    "creative": "You are a creative writer. Use vivid imagery, metaphors, and engaging storytelling.",
}

# Logging Setup
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("TeleBridge")

# Type Aliases
Update = t.Dict[str, t.Any]
Message = t.Dict[str, t.Any]
Chat = t.Dict[str, t.Any]


class TeleBridgeError(Exception):
    """Base exception for TeleBridge errors."""
    pass


class ConfigError(TeleBridgeError):
    """Configuration related errors."""
    pass


class TelegramAPIError(TeleBridgeError):
    """Errors related to Telegram Bot API interaction."""
    pass


class LocalAPIError(TeleBridgeError):
    """Errors related to Localhost API interaction."""
    pass


class TeleBridgeConfig:
    """Configuration manager for the CLI."""

    def __init__(self, args: argparse.Namespace):
        self.token = args.token or os.getenv(ENV_TELEGRAM_TOKEN)
        if not self.token:
            raise ConfigError(
                f"Telegram token not provided. Use --token or set {ENV_TELEGRAM_TOKEN}."
            )

        self.local_url = args.local_url or os.getenv(ENV_LOCAL_URL)
        if not self.local_url:
            raise ConfigError(
                f"Local URL not provided. Use --local-url or set {ENV_LOCAL_URL}."
            )

        self.model = args.model
        self.timeout = args.timeout
        self.verbose = args.verbose
        self.allowed_users = set(map(int, args.allowed_users.split(","))) if args.allowed_users else None
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("Configuration loaded successfully.")


class ConversationState:
    """Manages conversation history and active persona per chat."""

    def __init__(self):
        # Structure: { chat_id: { "persona": str, "history": list } }
        self._state: t.Dict[int, t.Dict[str, t.Any]] = {}
        self._lock = threading.Lock()

    def get_persona(self, chat_id: int) -> str:
        with self._lock:
            if chat_id not in self._state:
                self._state[chat_id] = {"persona": "default", "history": []}
            return PERSONAS.get(self._state[chat_id]["persona"], PERSONAS["default"])

    def set_persona(self, chat_id: int, persona_name: str) -> str:
        with self._lock:
            if chat_id not in self._state:
                self._state[chat_id] = {"persona": "default", "history": []}
            
            if persona_name in PERSONAS:
                self._state[chat_id]["persona"] = persona_name
                self._state[chat_id]["history"] = []  # Reset history on persona switch
                return f"Persona switched to '{persona_name}'. History cleared."
            else:
                available = ", ".join(PERSONAS.keys())
                return f"Unknown persona. Available: {available}"

    def get_history(self, chat_id: int) -> list:
        with self._lock:
            return self._state.get(chat_id, {}).get("history", [])

    def append_history(self, chat_id: int, role: str, content: str):
        with self._lock:
            if chat_id not in self._state:
                self._state[chat_id] = {"persona": "default", "history": []}
            self._state[chat_id]["history"].append({"role": role, "content": content})
            # Keep last 5 messages to save context window
            if len(self._state[chat_id]["history"]) > 5:
                self._state[chat_id]["history"] = self._state[chat_id]["history"][-5:]

    def reset_history(self, chat_id: int) -> str:
        with self._lock:
            if chat_id in self._state:
                self._state[chat_id]["history"] = []
            return "Conversation history reset."


class TelegramClient:
    """Handles interaction with the Telegram Bot API."""

    API_BASE = "https://api.telegram.org/bot"

    def __init__(self, token: str, timeout: int):
        self.token = token
        self.base_url = f"{self.API_BASE}{token}"
        self.timeout = timeout
        self.last_update_id = 0

    def get_updates(self) -> t.List[Update]:
        """Long-polling for new updates."""
        params = {
            "timeout": self.timeout,
            "offset": self.last_update_id + 1,
            "allowed_updates": ["message"]
        }
        try:
            response = requests.get(
                f"{self.base_url}/getUpdates", params=params, timeout=self.timeout + 5
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("ok"):
                raise TelegramAPIError(f"API Error: {data.get('description')}")
            
            updates = data.get("result", [])
            if updates:
                self.last_update_id = max(u["update_id"] for u in updates)
            
            return updates
        except requests.RequestException as e:
            logger.error(f"Network error fetching updates: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return []

    def send_message(self, chat_id: int, text: str) -> bool:
        """Sends a text message to a chat."""
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(f"{self.base_url}/sendMessage", json=payload)
            response.raise_for_status()
            data = response.json()
            if not data.get("ok"):
                logger.error(f"Failed to send message: {data.get('description')}")
                return False
            return True
        except requests.RequestException as e:
            logger.error(f"Network error sending message: {e}")
            return False

    def send_action(self, chat_id: int, action: str = "typing"):
        """Sends a chat action (e.g., typing)."""
        payload = {"chat_id": chat_id, "action": action}
        try:
            requests.post(f"{self.base_url}/sendChatAction", json=payload, timeout=5)
        except requests.RequestException:
            pass  # Ignore action failures


class LocalEngine:
    """Handles interaction with the Localhost API (e.g., Ollama)."""

    def __init__(self, url: str, model_name: str):
        self.url = url
        self.model_name = model_name

    def generate(self, prompt: str, history: t.List[t.Dict[str, str]]) -> str:
        """
        Sends a prompt to the local engine.
        Note: This implementation assumes a generic OpenAI-compatible or 
        Ollama-style POST request. It constructs the payload dynamically.
        """
        # Construct full context: System prompt + history + current prompt
        # Note: Depending on the local API, structure might vary. 
        # We attempt a standard chat completion format.
        
        payload = {
            "model": self.model_name,
            "stream": False,
            "prompt": prompt, # For Ollama /api/generate
            "options": {
                "temperature": 0.7
            }
        }
        
        # Try to detect if it's an OpenAI-compatible endpoint (common in local servers)
        # by checking if 'chat/completions' is in the path.
        if "chat" in self.url.lower():
             payload = {
                "model": self.model_name,
                "messages": history + [{"role": "user", "content": prompt}],
                "stream": False
            }

        try:
            logger.debug(f"Sending request to localhost: {json.dumps(payload, indent=2)}")
            response = requests.post(self.url, json=payload, timeout=120) # 2 min timeout for generation
            
            if response.status_code != HTTPStatus.OK:
                error_msg = response.text[:200]
                raise LocalAPIError(f"Local API returned {response.status_code}: {error_msg}")

            data = response.json()
            
            # Parse response based on common formats
            text_response = ""
            
            # Ollama format usually returns "response"
            if "response" in data:
                text_response = data["response"]
            # OpenAI format usually returns "choices" -> "message" -> "content"
            elif "choices" in data and len(data["choices"]) > 0:
                text_response = data["choices"][0].get("message", {}).get("content", "")
            else:
                # Fallback: try to find the most logical text field
                text_response = str(data)

            return text_response.strip()

        except requests.exceptions.Timeout:
            raise LocalAPIError("Request to local engine timed out.")
        except requests.exceptions.ConnectionError:
            raise LocalAPIError("Could not connect to local engine. Is it running?")
        except json.JSONDecodeError:
            raise LocalAPIError("Local engine returned invalid JSON.")


class Worker(threading.Thread):
    """Worker thread to handle processing of messages asynchronously."""

    def __init__(
        self, 
        queue: Queue, 
        tg_client: TelegramClient, 
        local_engine: LocalEngine, 
        state: ConversationState,
        allowed_users: t.Optional[t.Set[int]]
    ):
        super().__init__(daemon=True)
        self.queue = queue
        self.tg_client = tg_client
        self.local_engine = local_engine
        self.state = state
        self.allowed_users = allowed_users

    def run(self):
        logger.info("Worker thread started.")
        while True:
            task = self.queue.get()
            if task is None:  # Sentinel to exit
                break
            
            chat_id, text, message_id = task
            self.process_message(chat_id, text)
            self.queue.task_done()

    def process_message(self, chat_id: int, text: str):
        """Core logic to handle commands and generation."""
        
        # 1. Handle Commands
        if text.startswith("/"):
            command = text[1:].lower().split()[0]
            
            if command == "start":
                msg = (
                    "TeleBridge active.\n"
                    "Commands:\n"
                    "/code - Switch to coding persona\n"
                    "/essay - Switch to essay persona\n"
                    "/reset - Clear history\n"
                    "/default - Reset persona"
                )
                self.tg_client.send_message(chat_id, msg)
                return

            if command == "reset":
                reply = self.state.reset_history(chat_id)
                self.tg_client.send_message(chat_id, reply)
                return

            if command in PERSONAS:
                reply = self.state.set_persona(chat_id, command)
                self.tg_client.send_message(chat_id, reply)
                return
            
            # Fallback for unknown commands
            self.tg_client.send_message(chat_id, f"Unknown command: /{command}")
            return

        # 2. Handle Local Inference
        try:
            self.tg_client.send_action(chat_id, "typing")
            
            # Prepare context
            system_prompt = self.state.get_persona(chat_id)
            history = self.state.get_history(chat_id)
            
            # For simpler engines that don't support chat history via API,
            # we prepend history to the prompt string.
            full_prompt = f"{system_prompt}\n\n"
            for msg in history:
                role = msg['role'].upper()
                full_prompt += f"[{role}]: {msg['content']}\n"
            full_prompt += f"[USER]: {text}"

            # Call local engine
            response_text = self.local_engine.generate(full_prompt, history)
            
            # Save interaction
            self.state.append_history(chat_id, "user", text)
            self.state.append_history(chat_id, "assistant", response_text)
            
            # Send response (Telegram limit is 4096 chars)
            if len(response_text) > 4096:
                # Simple chunking logic
                for i in range(0, len(response_text), 4096):
                    self.tg_client.send_message(chat_id, response_text[i:i+4096])
            else:
                self.tg_client.send_message(chat_id, response_text)

        except TeleBridgeError as e:
            logger.error(f"Processing error for {chat_id}: {e}")
            self.tg_client.send_message(chat_id, f"Error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in worker")
            self.tg_client.send_message(chat_id, "Internal server error.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TeleBridge: Telegram to Localhost AI Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--token",
        help=f"Telegram Bot Token (overrides {ENV_TELEGRAM_TOKEN})"
    )
    parser.add_argument(
        "--local-url",
        help=f"Local API URL (overrides {ENV_LOCAL_URL}). Example: http://localhost:11434/api/generate"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to send to the local API (default: llama2)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_POLL_TIMEOUT,
        help="Long polling timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--allowed-users",
        help="Comma-separated list of Telegram User IDs allowed to use the bot. Leave empty for public (not recommended)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        config = TeleBridgeConfig(args)
        
        # Initialize components
        tg_client = TelegramClient(config.token, config.timeout)
        local_engine = LocalEngine(config.local_url, config.model)
        state = ConversationState()
        
        # Task Queue
        task_queue = Queue()
        
        # Start Worker Thread
        worker = Worker(task_queue, tg_client, local_engine, state, config.allowed_users)
        worker.start()
        
        logger.info("TeleBridge started. Listening for messages...")
        logger.info(f"Target Local URL: {config.local_url}")
        logger.info(f"Target Model: {config.model}")

        # Main Loop (Long Polling)
        while True:
            updates = tg_client.get_updates()
            
            for update in updates:
                if "message" not in update:
                    continue
                
                message = update["message"]
                chat = message.get("chat")
                text = message.get("text")
                
                if not chat or not text:
                    continue
                
                chat_id = chat["id"]
                user_id = chat.get("id") # In private chats, chat_id == user_id
                
                # Security Check
                if config.allowed_users and user_id not in config.allowed_users:
                    logger.warning(f"Unauthorized access attempt from {user_id}")
                    tg_client.send_message(chat_id, "⛔ Unauthorized access.")
                    continue

                logger.info(f"Received message from {chat_id}: {text[:50]}...")
                
                # Enqueue task for worker thread
                task_queue.put((chat_id, text, message["message_id"]))

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        task_queue.put(None)  # Stop worker
        worker.join()
        sys.exit(0)
    except ConfigError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error in main loop")
        sys.exit(1)


if __name__ == "__main__":
    main()