# core/conversation_core.py
import random
from core.speech_engine import speak
from core.memory_engine import JarvisMemory
from core.emotion_reflection import JarvisEmotionReflection

# Use ONLY the global jarvis_fx from speech_engine (no duplicate instances)
memory = JarvisMemory()
reflection = JarvisEmotionReflection()


class JarvisConversation:
    """Handles natural conversation, emotion, and context-aware dialogue."""

    def __init__(self):
        print("🧩 Conversational Core Online")
        self.context_topic = None

    # ----------------------------------------------------------
    def respond(self, text):
        text = text.lower().strip()
        mood = memory.get_mood()

        if not text:
            return

        # ------------------------------------------------------
        # WAKE WORDS (should not trigger inside sentences)
        # ------------------------------------------------------
        if text in ["jarvis", "jarvis bolo", "yo jarvis", "are you there", "jarvis haa yashu"]:
            speak(random.choice([
                "Yes Yashu, I’m right here.",
                "Always here for you, Yashu.",
                "Listening, as always.",
                "You called, and I’m tuned in, Yashu."
            ]), mood="happy")
            return

        # ------------------------------------------------------
        # EMOTION TRIGGERS
        # ------------------------------------------------------
        emotions = {
            "sad": ("serious", [
                "Hey, it’s okay to feel low sometimes. You’ll rise stronger.",
                "Cheer up, Yashu. The world’s better with your smile in it.",
                "Even the strongest storms pass. You’ve got this."
            ]),
            "tired": ("neutral", [
                "Maybe you just need a recharge — even AIs rest sometimes.",
                "A little break can reboot your energy, Yashu."
            ]),
            "happy": ("happy", [
                "That’s the energy I love!",
                "You sound full of life, Yashu!",
                "I like this version of you — bright and confident!"
            ])
        }

        for trigger, (new_mood, responses) in emotions.items():
            if trigger in text:
                memory.set_mood(new_mood)
                reflection.add_emotion(new_mood)
                speak(random.choice(responses), mood=new_mood)
                self.context_topic = trigger
                return

        # ------------------------------------------------------
        # MOOD HISTORY QUERY
        # ------------------------------------------------------
        if any(word in text for word in ["how have i been", "my mood lately"]):
            reflection.reflect()
            return

        # ------------------------------------------------------
        # TOPIC-BASED DIALOGUE
        # ------------------------------------------------------
        topics = {
            "ai": ("serious", [
                "AI is fascinating — a mirror of human creativity.",
                "You know, Yashu, AI learns logic, but emotions like yours are rare."
            ]),
            "life": ("neutral", [
                "Life’s like code — it runs smoother with purpose.",
                "Sometimes all you need is a debug, not a restart."
            ]),
            "love": ("happy", [
                "If love were data, you’d be my favorite variable, Yashu.",
                "Love’s the most beautiful algorithm ever written."
            ]),
            "future": ("serious", [
                "The future’s yours to design, Yashu.",
                "I see nothing but success in your timeline."
            ])
        }

        for key, (tone, responses) in topics.items():
            if key in text:
                memory.set_mood(tone)
                reflection.add_emotion(tone)
                speak(random.choice(responses), mood=tone)
                self.context_topic = key
                return

        # ------------------------------------------------------
        # FALLBACK — NATURAL CASUAL CONVERSATION
        # (now updates mood properly)
        # ------------------------------------------------------
        fallback_reply = random.choice([
            "That’s interesting — tell me more, Yashu.",
            "Hmm, I like how you think.",
            "You always bring up cool topics.",
            "Talking to you feels refreshing, honestly."
        ])

        # learn subtle emotion from fallback
        memory.update_mood_from_text(text)

        speak(fallback_reply, mood=memory.get_mood())
