# 🤖 JARVIS  
### An Emotion-Aware, Voice-First, Offline-First AI Personal Assistant

> **JARVIS is not a chatbot.**  
> It is a cognitive, emotionally-adaptive AI system designed to understand, assist, and evolve with its user — not just respond to commands.

---

## 🌌 Vision

JARVIS is a **next-generation personal AI assistant** built around **human-centric intelligence**, not rigid command automation.

Unlike traditional voice assistants, JARVIS:
- Understands intent, emotion, and context
- Maintains continuity across conversations
- Operates offline-first using local AI models
- Acts as a trusted digital companion, not just a tool

This project is engineered with **research-grade architecture**, **modular AI pipelines**, and **real system-level control**, targeting an **OpenAI-level design philosophy**.

---

## 🧠 Core Design Philosophy

| Principle | Description |
|---------|------------|
| Voice-First Interaction | Natural, always-on voice communication with wake-word detection |
| Emotion Awareness | Text-based emotion detection with memory-driven mood fusion |
| Offline-First AI | Local LLMs via Ollama, zero cloud dependency |
| Human-like Flow | Multi-turn conversations, follow-ups, pauses, hesitation handling |
| Modular Intelligence | Clear separation of perception, cognition, decision, and action |
| Safety & Control | Intent validation before executing system actions |

---

## 🏗️ System Architecture (High Level)

```text
Microphone
   ↓
Background Listener
   ↓
Intent Parser (NLP)
   ↓
JARVIS Cognitive Core
(AI Reasoning + Emotion Engine + Memory)
   ↓
Command Handler
(System / Web / Media / Automation)
   ↓
Speech Engine (TTS)
```

---

## 🚀 Key Features

### 🎧 Voice & Interaction
- Always-on background listener
- Wake words: Hey Jarvis, Ok Jarvis
- Handles silence, partial speech, hesitation
- Human-like thinking feedback

### 🧠 AI Intelligence
- Local LLMs via Ollama
- Automatic model detection and fallback
- Fully offline operation
- Short, voice-optimized replies

### 😊 Emotion & Personality Engine
- Text-based emotion inference
- Mood fusion using memory and history
- Adaptive personality (calm, supportive, energetic)

### 🧠 Persistent Memory
- Long-term memory stored in memory.json
- Remembers personal facts and preferences
- Tracks emotional trends
- Topic continuity across sessions

### 🖥️ System Control
- Application launch and control
- Volume, brightness, mute, lock
- Window management
- Screenshot capture
- Battery, time, and date awareness

### 🌐 Web & Automation
- YouTube automation (Selenium)
- WhatsApp Web automation
- Google search automation

### 📄 Intelligent Content Handling
- Document reading (PDF, DOCX, TXT)
- Document summarization
- Video summarization
- Local music playback
- Online music streaming

---

## 🧩 Project Structure

```text
Jarvis/
├── main.py                  # Main entry point (UI + backend)
├── start_jarvis.py          # Background-only execution
├── core/
│   ├── ai_chat.py
│   ├── brain.py
│   ├── background_listener.py
│   ├── command_handler.py
│   ├── conversation_core.py
│   ├── intent_parser.py
│   ├── nlp_engine.py
│   ├── memory_engine.py
│   ├── context.py
│   ├── state.py
│   ├── speech_engine.py
│   ├── desktop_control.py
│   ├── document_reader.py
│   ├── video_reader.py
│   ├── music_player.py
│   ├── music_stream.py
│   ├── youtube_driver.py
│   └── whatsapp_selenium.py
├── config/
│   ├── memory.json
│   ├── settings.json
│   └── nlp_history.txt
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Category | Technology |
|--------|------------|
| Language | Python |
| AI / LLM | Local LLMs (Ollama) |
| Speech | SpeechRecognition, TTS |
| Automation | Selenium, PyAutoGUI |
| UI | PyQt5 |
| NLP | Custom intent & emotion engine |
| Platform | Windows |

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

Background-only mode:
```bash
python start_jarvis.py
```

---

## 🎯 Use Cases
- Personal AI companion
- Productivity assistant
- Emotional wellness support
- Research platform for human-centric AI

---

## 🔮 Future Advancements

JARVIS is designed as an evolving, human-centric AI system.  
Future development focuses on deeper personalization, emotional intelligence, and autonomous decision-making.

- **Identity-Aware Intelligence** – Voice-based owner recognition and command authorization  
- **Multimodal Emotion Understanding** – Emotion detection using text, voice, and facial cues  
- **Habit & Behavior Learning** – Long-term tracking of routines, stress, and productivity patterns  
- **Autonomous Decision Engine** – Context-aware interventions with user-overridable control  
- **Cognitive Memory Expansion** – Structured memory, contextual recall, and intelligent forgetting  
- **Plugin-Based Skill System** – Extensible architecture for adding new capabilities  
- **Explainable AI** – Transparent reasoning and self-evaluation for critical decisions  
- **Cross-Device Continuity** – Secure memory and context synchronization across devices  

> *“JARVIS is not built to replace humans,  
but to stand beside one — consistently, intelligently, and responsibly.”*

---

## 👨‍💻 Author

**Yash Kadam**  
AI & ML Engineer | Builder of Human-Centric, Emotion-Aware AI Systems  

> “I didn’t want to build a chatbot.  
> I wanted to build someone who stays.”
