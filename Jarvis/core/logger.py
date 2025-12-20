import logging

logging.basicConfig(
    filename="jarvis_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("JARVIS")
