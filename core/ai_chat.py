# core/ai_chat.py — FINAL UPGRADED VERSION (Dynamic Model + Memory Fix)

"""
AI Chat Brain for Jarvis
- Automatically detects installed Ollama models.
- Falls back to conversation_core if Ollama is offline.
- Injects memory + mood + last topic correctly.
- Optimized for Voice (Short answers, No Markdown).
"""

import json
import time
import requests

# -------------------------------------------
# Optional HTTP client for Ollama raw requests
# -------------------------------------------
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

# -------------------------------------------
# Optional direct Ollama Python package
# -------------------------------------------
try:
    import ollama
    _HAS_OLLAMA_PKG = True
except ImportError:
    ollama = None
    _HAS_OLLAMA_PKG = False

# -------------------------------------------
# Local fallback convo
# -------------------------------------------
try:
    from core.conversation_core import JarvisConversation
    _HAS_CONV = True
except ImportError:
    JarvisConversation = None
    _HAS_CONV = False

# -------------------------------------------
# Memory + State
# -------------------------------------------
try:
    from core.context import memory
except ImportError:
    memory = None

import core.state as state


# ============================================================
#   OLLAMA CLIENT (Smart Model Selection)
# ============================================================
class OllamaClient:
    def __init__(self, default_model="llama3"):
        self.base_url = "http://localhost:11434"
        self.model = self._find_best_model(default_model)
        print(f"🧠 AI Brain linked to model: {self.model}")

    def _find_best_model(self, preferred):
        """Checks available models and picks the best one installed."""
        try:
            # 1. Try to fetch list of installed models
            response = requests.get(f"{self.base_url}/api/tags", timeout=1.0)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                
                # Check for preferred specific match
                for m in models:
                    if preferred in m: return m
                
                # Check for known good fallback models
                for fallback in ["llama3.1", "llama3", "mistral", "phi3", "gemma", "llama2"]:
                    for m in models:
                        if fallback in m: return m
                
                # If any model exists, return the first one
                if models: return models[0]
                
        except Exception:
            pass
        
        # Default blind return if check fails
        return preferred

    def available(self):
        """Check if Ollama server is actually running."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            return r.status_code == 200
        except:
            return False

    def ask(self, system_prompt, user_prompt):
        """Send request to Ollama."""
        # Method A: Python Library (Preferred)
        if _HAS_OLLAMA_PKG:
            try:
                out = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return out.get("message", {}).get("content", "").strip()
            except Exception as e:
                # print(f"Ollama Pkg Error: {e}") # Debug only
                pass

        # Method B: HTTPX / Raw Request (Backup)
        try:
            payload = {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            # Use requests if httpx missing, or httpx if available
            if _HAS_HTTPX:
                r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=20.0)
                data = r.json()
            else:
                r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=20.0)
                data = r.json()

            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"⚠️ AI Connection Error: {e}")
            return None


# ============================================================
#   LOCAL FALLBACK (If Brain is Offline)
# ============================================================
class LocalFallback:
    def __init__(self):
        self.conv = JarvisConversation() if _HAS_CONV else None

    def ask(self, prompt):
        if not self.conv:
            return "Systems offline. I can't think right now."
        try:
            r = self.conv.respond(prompt)
            return r or "I didn't catch that."
        except:
            return "I couldn't process that locally."


# ============================================================
#   AI CHAT BRAIN (MAIN LOGIC)
# ============================================================
class AIChatBrain:
    def __init__(self):
        # We start with a generic name, the client will resolve the specific tag
        self.ollama = OllamaClient(default_model="llama3")
        self.fallback = LocalFallback()

    # ---------------------------------------------------------
    # Build the personality + memory prompt
    # ---------------------------------------------------------
    def _build_system_prompt(self):
        # 1. Get Mood
        mood = "neutral"
        if memory:
            mood = memory.get_mood()
        
        # 2. Get Last Topic
        last_topic = getattr(state, "LAST_TOPIC", "general chat")

        # 3. Get Memories (FIXED: accessing dictionary directly)
        mem_text = "No prior facts."
        if memory and hasattr(memory, "memory"):
            facts_dict = memory.memory.get("facts", {})
            if facts_dict:
                # Format: "- User's name is Yash"
                fact_list = [f"- {k} is {v}" for k, v in facts_dict.items()]
                mem_text = "\n".join(fact_list)

        return f"""
You are Jarvis, a highly intelligent, witty, and loyal AI assistant.
User: Yash.

YOUR STATE:
- Current Mood: {mood}
- Previous Topic: {last_topic}

MEMORY OF USER:
{mem_text}

INSTRUCTIONS:
1. RESPONSE FORMAT: You are talking via Text-to-Speech.
   - Keep answers SHORT (1-2 sentences max) unless asked for details.
   - DO NOT use Markdown (no **bold**, no # headers).
   - DO NOT use code blocks unless explicitly asked to write code.
   - Use natural, conversational English (occasional Hinglish is okay).

2. PERSONALITY:
   - If Yash is sad, be comforting and gentle.
   - If Yash is happy, be energetic.
   - If asked to do a task, say "On it" or "Sure thing".
   - Never say "As an AI language model". You are Jarvis.

3. GOAL: Be a helpful friend, not just a robot.
"""

    # ---------------------------------------------------------
    # Main Public Function: ASK
    # ---------------------------------------------------------
    def ask(self, prompt: str):
        if not prompt:
            return "I'm listening."

        # 1. Build context
        system_prompt = self._build_system_prompt()

        # 2. Try Ollama (The Brain)
        if self.ollama.available():
            ans = self.ollama.ask(system_prompt, prompt)
            if ans:
                # Update global topic state if successful
                try:
                    state.LAST_TOPIC = prompt
                except: pass
                return ans.strip() if ans else None


        # 3. Failover to Local (Scripted)
        print("⚠️ Ollama offline/unreachable. Using local fallback.")
        return self.fallback.ask(prompt)


# ============================================================
# Export singleton
# ============================================================
ai_chat_brain = AIChatBrain()