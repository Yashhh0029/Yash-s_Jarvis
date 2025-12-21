# core/context.py
"""
Global shared context for Jarvis
Single source of truth for memory and state access
"""

from core.memory_engine import JarvisMemory
import core.state as state

memory = JarvisMemory()

def set_topic(topic):
    if topic:
        state.LAST_TOPIC = topic

def get_topic():
    return getattr(state, "LAST_TOPIC", None)

def set_mood(mood):
    if mood:
        state.JARVIS_MOOD = mood

def get_mood():
    return getattr(state, "JARVIS_MOOD", "neutral")

# --------------------------------------------------
# LAST ACTION TRACKING (YouTube / multi-turn logic)
# --------------------------------------------------

_LAST_ACTION = None

def set_last_action(action):
    """
    Store last high-level action like:
    'youtube_search'
    """
    global _LAST_ACTION
    _LAST_ACTION = action

def get_last_action():
    """
    Retrieve last stored action
    """
    return _LAST_ACTION
