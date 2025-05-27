import os
import subprocess

# Language codes and their base audio URLs
languages = {
    # "fr": "French",
    # "de": "German",
    # "ca": "Catalan",
    # "es": "Spanish",
    # "pl": "Polish",
    # "nl": "Dutch",
    # "fa": "Persian",
    # "rw": "Kinyarwada",
    # "it": "Italian"
    "tr": "Turkish",
    # "ar": "Arabic"
}

audio_base_url = "https://mswc.mlcommons-storage.org/audio"
splits_base_url = "https://mswc.mlcommons-storage.org/splits"

for lang in languages:
    print(f"\n=== Processing {lang} ({languages[lang]}) ===")
    
    # Create language folder like: ../data/{lang}, for example Keyword-MLP-ISSAI/data/ar
    lang_dir = os.path.join("data", lang)
    os.makedirs(lang_dir, exist_ok=True)
    os.chdir(lang_dir)

    # Download audio and splits tar.gz files
    audio_url = f"{audio_base_url}/{lang}.tar.gz"
    splits_url = f"{splits_base_url}/{lang}.tar.gz"

    print(f"Downloading: {audio_url}")
    subprocess.run(["wget", audio_url])

    print(f"Downloading: {splits_url}")
    subprocess.run(["wget", splits_url])

    # Unarchive both
    print(f"Extracting: {lang}.tar.gz")
    subprocess.run(["tar", "-xvzf", f"{lang}.tar.gz"])

    print(f"Extracting: {lang}.tar.gz.1 (split)")
    subprocess.run(["tar", "-xvzf", f"{lang}.tar.gz.1"])

    # Go back to parent directory
    os.chdir("..")