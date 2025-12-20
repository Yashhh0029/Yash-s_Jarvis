# core/conversation_core.py
"""
Hybrid conversational core for Jarvis (Hybrid Mode - natural + consistent).
"""

import random
import re
from typing import Optional
from collections import deque

from core.brain import brain
from core.context import memory, set_topic, get_topic
from core.emotion_reflection import JarvisEmotionReflection

reflection = JarvisEmotionReflection()

import core.nlp_engine as nlp


def _word_bound_search(word_list, text):
    for w in word_list:
        if re.search(rf'\b{re.escape(w)}\b', text, flags=re.IGNORECASE):
            return True
    return False


class JarvisConversation:

    def __init__(self):
        print("🧩 Conversational Core Online (Hybrid Mode)")
        self.last_topic = None
        self.last_response = ""
        self.recent_fallbacks = deque(maxlen=8)
        self._recent_user_queries = deque(maxlen=12)

    def _estimate_sentiment(self, text: Optional[str]) -> str:
        if not text:
            return "neutral"

        t = text.lower()

        sad = ["sad", "low", "down", "hurt", "upset", "empty", "broken", "depressed", "lonely", "tear"]
        happy = ["happy", "great", "awesome", "nice", "good", "fantastic", "amazing", "glad", "yay", "excited"]
        angry = ["angry", "mad", "pissed", "furious", "hate", "annoyed"]
        anxious = ["scared", "worried", "anxious", "panic", "stressed", "stress", "overthinking", "nervous"]
        bored = ["bored", "meh", "boring", "idle"]

        score = 0
        if _word_bound_search(sad, t):
            score -= 2
        if _word_bound_search(angry, t):
            score -= 3
        if _word_bound_search(anxious, t):
            score -= 2
        if _word_bound_search(happy, t):
            score += 3
        if _word_bound_search(bored, t):
            score -= 1

        if re.search(r"\b(no|not|n't|never)\b", t):
            if _word_bound_search(happy, t):
                score -= 2
            if _word_bound_search(sad, t):
                score += 1

        if score >= 2:
            return "happy"
        if score <= -2:
            if _word_bound_search(angry, t):
                return "alert"
            return "serious"
        return "neutral"

    def _detect_topic(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None

        t = text.lower()
        topic_map = {
            "ai": ["ai", "artificial intelligence", "machine learning", "deep learning"],
            "java": ["java", "jvm", "spring"],
            "daa": ["daa", "dynamic programming", "algorithms", "graphs"],
            "python": ["python", "py"],
            "graphics": ["graphics", "opengl", "projection", "3d"],
            "dbms": ["dbms", "database", "sql", "mysql", "postgres", "oracle"],
            "blockchain": ["blockchain", "ethereum", "smart contract"],
            "gesture": ["gesture", "hand gesture", "gesture recognition"],
            "emotion": ["emotion", "mood", "feeling"],
            "life": ["life", "future", "career"],
            "love": ["love", "relationship", "gf", "bf", "crush"],
        }

        for key, aliases in topic_map.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", t):
                    return key

        m = re.search(r"\b([a-zA-Z]{3,20})\b", t)
        return m.group(1) if m else None

    def _continue_topic(self) -> str:
        if not self.last_topic:
            return "Continue what, Yashu? Remind me the topic and I'll follow up."

        base_templates = {
            "ai": "AI improves when you iterate on datasets and objectives. Want a small example or a project idea?",
            "java": "Java is excellent for large apps — focus on design patterns and testing. Want a sample structure?",
            "life": "Small consistent habits beat sudden bursts. Which habit should we plan first?",
            "love": "Communication and patience are key. Want a gentle script to start a conversation?",
            "daa": "For DAA, practice time/space trade-offs with real problems — want 3 practice problems?",
            "dbms": "Normalization and indexing matter. Want a quick explanation of normalization levels?"
        }

        return base_templates.get(
            self.last_topic,
            f"Let's explore more about {self.last_topic}. Which part interests you?"
        )

    def respond(self, text: Optional[str]) -> str:
        if not text:
            return "Yes Yashu? I'm listening."

        raw = text.strip()
        t = raw.lower()

        try:
            self._recent_user_queries.append(raw)
        except Exception:
            pass

        try:
            nlp.learn_async(raw)
        except Exception:
            pass

        try:
            mood = self._estimate_sentiment(t)
            memory.set_mood(mood)
            reflection.add_emotion(mood)
        except Exception:
            mood = memory.get_mood() or "neutral"

        if t in ("jarvis", "hey jarvis", "are you there", "yo jarvis", "jarvis bolo", "jarvis haan"):
            try:
                return brain.generate_wakeup_line(mood=memory.get_mood(), last_topic=self.last_topic)
            except Exception:
                return "Yes Yashu, I am here."

        try:
            if re.search(r"\b(i am|i'm|i feel|feeling)\b.*\b(sad|low|hurt|empty|depressed|lonely)\b", t):
                return brain.generate_emotional_support("sad", mood)
            if re.search(r"\b(i am|i'm|i feel|feeling)\b.*\b(happy|great|good|awesome|excited)\b", t):
                return brain.generate_emotional_support("happy", mood)
        except Exception:
            pass

        if any(w in t for w in ("continue", "more", "keep going", "tell me more")):
            return self._continue_topic()

        if re.search(r"\b(what|why|how|explain|help|define)\b", t):
            topic = self._detect_topic(t)
            self.last_topic = topic
            set_topic(topic)
            try:
                memory.update_topic(topic)
            except Exception:
                pass
            try:
                return brain.answer_question(t, topic, mood)
            except Exception:
                return f"I can explain {topic or 'that'} — short summary or a detailed explanation?"

        if any(w in t for w in ("open", "launch", "play", "type", "search", "screenshot", "volume", "brightness", "notepad", "whatsapp")):
            topic = self._detect_topic(t)
            self.last_topic = topic
            set_topic(topic)
            try:
                memory.update_topic(topic)
            except Exception:
                pass
            return random.choice([
                "On it, Yash. Doing that now.",
                "Got the command — executing.",
                "Alright — I'll take care of that."
            ])

        fallback_pool = [
            "Hmm… interesting. Want to explore that?",
            "Tell me more — I’m following you.",
            "You always think differently. What’s the next part?",
            "I can dive deeper into that if you want.",
            "Want a breakdown, a summary, or a story version?"
        ]

        reply = random.choice(fallback_pool)
        if self.recent_fallbacks and reply == self.recent_fallbacks[-1]:
            alt = [r for r in fallback_pool if r != reply]
            if alt:
                reply = random.choice(alt)

        try:
            recent_same = sum(1 for q in self._recent_user_queries if q.lower() == raw.lower())
            if recent_same >= 2:
                reply = "Seems like you want a clear answer. Do you want a short summary or a step-by-step example?"
        except Exception:
            pass

        try:
            self.last_topic = self._detect_topic(t)
            set_topic(self.last_topic)
            memory.update_topic(self.last_topic)
        except Exception:
            pass

        try:
            self.recent_fallbacks.append(reply)
        except Exception:
            pass

        self.last_response = reply
        return reply
