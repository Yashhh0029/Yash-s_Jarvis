import os

# ======================================================
# CONFIG (MATCHES YOUR STRUCTURE EXACTLY)
# ======================================================
PROJECT_ROOT = "Jarvis"   # target folder
OUTPUT_FILE = "JARVIS_FULL_PROJECT.py"

INCLUDE_EXTENSIONS = {".py", ".json", ".txt", ".md"}

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    "sounds",
    "models",
    "faces",
    "face_data",
    "data"
}

EXCLUDE_FILES = {
    OUTPUT_FILE,
    "requirements.txt"
}

# ======================================================
# LOGIC
# ======================================================
def combine_project():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("# ==============================================================\n")
        out.write("#  JARVIS – FULL COMBINED PROJECT SOURCE CODE\n")
        out.write("#  Auto-generated from /Jarvis directory\n")
        out.write("# ==============================================================\n\n")

        for root, dirs, files in os.walk(PROJECT_ROOT):
            # remove excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in sorted(files):
                if file in EXCLUDE_FILES:
                    continue

                ext = os.path.splitext(file)[1]
                if ext not in INCLUDE_EXTENSIONS:
                    continue

                file_path = os.path.join(root, file)

                out.write("\n\n")
                out.write("#" * 110 + "\n")
                out.write(f"# FILE: {file_path}\n")
                out.write("#" * 110 + "\n\n")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        out.write(f.read())
                except Exception as e:
                    out.write(f"# ERROR READING FILE: {e}\n")

    print(f"\n✅ SUCCESS: Combined JARVIS code saved as → {OUTPUT_FILE}\n")


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    combine_project()
