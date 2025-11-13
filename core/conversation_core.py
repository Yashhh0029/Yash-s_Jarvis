# ============================================================
#        JARVIS CONVERSATION CORE — MERGED & UPGRADED
# ============================================================

import random
import re
from core.speech_engine import speak
from core.memory_engine import JarvisMemory
from core.emotion_reflection import JarvisEmotionReflection

memory = JarvisMemory()
reflection = JarvisEmotionReflection()


class JarvisConversation:
    """
    FINAL MERGED VERSION:
    ✔ Keeps all your old features
    ✔ Adds continuation support
    ✔ Adds better natural answers
    ✔ Adds topic detection & fallback logic
    ✔ NO conflict with command_handler
    ✔ NO removal of “Yashu” emotional lines
    """

    def __init__(self):
        print("🧩 Conversational Core Online")
        self.last_topic = None
        self.last_response = ""
        self.last_was_long = False

    # ============================================================
    # SENTIMENT — lightweight mood update
    # ============================================================
    def _estimate_sentiment(self, text):
        text = text.lower()

        if any(w in text for w in ["sad", "depressed", "low", "hurt", "broken"]):
            return "serious"
        if any(w in text for w in ["happy", "great", "awesome", "good"]):
            return "happy"

        return "neutral"

    # ============================================================
    # TOPIC DETECTION (for follow-up & continue feature)
    # ============================================================
    def _detect_topic(self, text):
        important = [
            "ai", "machine learning", "java", "python", "daa",
            "life", "love", "future", "emotion", "blockchain",
            "cryptography", "graphics", "os", "database",
            "data structures"
        ]
        for t in important:
            if t in text:
                return t

        # fallback → detect first noun-like chunk
        m = re.search(r"\b([a-zA-Z ]+)\b", text)
        return m.group(1).strip() if m else None

    # ============================================================
    # CONTINUATION LOGIC
    # ============================================================
    def _continue_topic(self):
        if not self.last_topic:
            reply = "Continue what, Yashu? Remind me the topic."
            speak(reply, mood="neutral")
            return reply

        expansions = {
            "ai": "AI becomes more powerful when data variety increases. Creativity emerges from patterns.",
            "java": "Java focuses on stability and portability — JVM is its real superpower.",
            "daa": "Algorithm efficiency saves time exponentially — even tiny optimizations matter.",
            "life": "Life isn't about speed, it's about intention and clarity.",
            "love": "Love isn't logic — it's chemistry, timing, and understanding.",
            "future": "Your dedication determines your future graph — and yours is rising fast."
        }

        reply = expansions.get(
            self.last_topic,
            f"Let’s go deeper into {self.last_topic}. It's more interesting than it looks."
        )

        self.last_was_long = len(reply) > 120
        speak(reply, mood=memory.get_mood())
        return reply

    # ============================================================
    # MAIN RESPONSE FUNCTION
    # ============================================================
    def respond(self, text):
        if not text:
            return

        text = text.lower().strip()

        # Estimate mood from user text
        mood = self._estimate_sentiment(text)
        memory.set_mood(mood)

        # ----------------------------------------------------------
        # WAKEWORD CASUAL
        # ----------------------------------------------------------
        if text in ["jarvis", "jarvis bolo", "yo jarvis", "are you there", "jarvis haa yashu"]:
            speak(random.choice([
                "Yes Yashu, I’m right here.",
                "Always here for you, Yashu.",
                "Listening, as always.",
                "You called, and I’m tuned in, Yashu."
            ]), mood="happy")
            return

        # ----------------------------------------------------------
        # USER-EMOTION TRIGGERS (your old logic preserved)
        # ----------------------------------------------------------
        emotion_presets = {
            "sad": ("serious", [
                "Hey… it’s okay to feel low. I’m right here with you.",
                "Cheer up, Yashu. Your smile lights up more than you think.",
                "Storms pass, and so will this feeling."
            ]),
            "tired": ("neutral", [
                "Maybe you just need a break. Even AIs cool down sometimes.",
                "Rest isn’t weakness, it's recharge."
            ]),
            "happy": ("happy", [
                "That's the energy I love!",
                "You sound bright today, Yashu!",
                "This vibe suits you!"
            ])
        }

        for word, (emo, lines) in emotion_presets.items():
            if word in text:
                memory.set_mood(emo)
                reflection.add_emotion(emo)
                speak(random.choice(lines), mood=emo)
                self.last_topic = word
                return

        # ----------------------------------------------------------
        # USER ASKING ABOUT THEIR MOOD HISTORY
        # ----------------------------------------------------------
        if any(w in text for w in ["how have i been", "my mood lately", "was i okay", "my feelings history"]):
            reflection.reflect()
            return

        # ----------------------------------------------------------
        # TOPIC RESPONSES (your old ones preserved)
        # ----------------------------------------------------------
        predefined_topics = {
            "ai": ("serious", [
                "AI is fascinating — a mirror of human creativity.",
                "AI learns logic, but emotions like yours are rare."
            ]),
            "life": ("neutral", [
                "Life is like code — it runs smoother with purpose.",
                "Sometimes debugging life works better than restarting it."
            ]),
            "love": ("happy", [
                "If love were data, you'd be my favorite variable, Yashu.",
                "Love is the most beautiful algorithm humans ever wrote."
            ]),
            "future": ("serious", [
                "Your future looks bright with your mindset.",
                "You're on a strong path, Yashu. Momentum matters."
            ]),
        }

        for key, (tone, lines) in predefined_topics.items():
            if key in text:
                memory.set_mood(tone)
                reflection.add_emotion(tone)
                speak(random.choice(lines), mood=tone)
                self.last_topic = key
                return

        # ----------------------------------------------------------
        # CONTINUE FEATURE
        # ----------------------------------------------------------
        if any(w in text for w in ["continue", "more", "keep going", "tell me more"]):
            return self._continue_topic()

        # ----------------------------------------------------------
        # QUESTION-BASED SMART ANSWERS
        # ----------------------------------------------------------
        if "what" in text or "why" in text or "how" in text:
            topic = self._detect_topic(text)
            self.last_topic = topic

            explanations = {
                "ai": "Artificial Intelligence helps machines learn patterns and make decisions.",
                "java": "Java is reliable and portable, runs on JVM anywhere.",
                "daa": "DAA tells you how fast or efficient an algorithm truly is.",
                "life": "Life feels complex, but clarity comes when you slow down.",
            }

            reply = explanations.get(topic, f"{topic} is interesting — want a deeper explanation?")
            speak(reply, mood=mood)
            self.last_was_long = len(reply) > 120
            return reply

        # ----------------------------------------------------------
        # NATURAL FALLBACK (soft, casual and non-robotic)
        # ----------------------------------------------------------
        fallback = random.choice([
            "That’s interesting — tell me more, Yashu.",
            "Hmm, I like how you're thinking.",
            "You always bring up unique thoughts.",
            "Talking with you feels refreshing."
        ])

        memory.update_mood_from_text(text)
        speak(fallback, mood=memory.get_mood())
        self.last_topic = self._detect_topic(text)
        return fallback
