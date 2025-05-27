import subprocess
import os

# List of language codes to process
languages = [
    # 'tt',
    'tr',
    # 'ru'
    # 'ar'
    # "fr"
    # "de",
    # "ca",
    # "es",
    # "pl",
    # "nl",
    # "fa",
    # "rw",
    # "it",  
]

# Absolute or relative path to full_process.py
FULL_PROCESS_SCRIPT = "utils/process_mswc.py"

for lang in languages:
    print(f"\n=== Running process_mswc.py for language: {lang} ===")

    # Run full_process.py with 'language' passed via environment variable
    result = subprocess.run(
        ["python", FULL_PROCESS_SCRIPT],
        env={**os.environ, "language": lang},
    )

    # Optional: check if it failed
    if result.returncode != 0:
        print(f"❌ Error processing language: {lang}")
    else:
        print(f"✅ Successfully processed: {lang}")