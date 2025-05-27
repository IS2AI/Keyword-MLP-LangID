import os
import shutil
import pandas as pd
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

language = os.environ.get("language", "fr")

# Paths
csv_file_path = 'utils/mswc_overlapping_commands.csv'
clips_dir_path = f'data/{language}/{language}/clips'

# Text files to update
testing_list = f'data/{language}/{language}/clips/testing_list.txt'
validation_list = f'data/{language}/{language}/clips/validation_list.txt'
training_list = f'data/{language}/{language}/clips/training_list.txt'
text_files = [testing_list, validation_list, training_list]

# Path to the splits CSV file (e.g., ar_splits.csv, tr_splits.csv)
splits_csv_path = f'data/{language}/{language}_splits.csv'

# Read the splits CSV file
splits_df = pd.read_csv(splits_csv_path)

# Counters for debug
test_count = 0
val_count = 0
train_count = 0

# Open output files
with open(testing_list, 'w') as test_file, \
     open(validation_list, 'w') as val_file, \
     open(training_list, 'w') as train_file:
    for index, row in tqdm(splits_df.iterrows(), total=len(splits_df), desc='Initial split processing'):
        file_path = os.path.join(clips_dir_path, row['LINK'])
        set_type = str(row['SET']).strip().lower()
        if os.path.exists(file_path):
            if set_type == 'test':
                test_file.write(f"{row['LINK']}\n")
                test_count += 1
            elif set_type == 'dev':
                val_file.write(f"{row['LINK']}\n")
                val_count += 1
            elif set_type == 'train':
                train_file.write(f"{row['LINK']}\n")
                train_count += 1

# Read the CSV file for command mapping, use -777 as NaN and do not treat NaN and null as NaN
commands_df = pd.read_csv(csv_file_path, na_values=['-777'], keep_default_na=False)

# Extract language-specific commands
lang_commands = commands_df[language].dropna().unique()

# print(lang_commands)

# List all directories in the clips folder
all_folders = [f for f in os.listdir(clips_dir_path) if os.path.isdir(os.path.join(clips_dir_path, f))]

# Determine folders to delete
folders_to_delete = [folder for folder in all_folders if folder not in lang_commands]

# Delete folders that are not in the language commands list
for folder in folders_to_delete:
    folder_path = os.path.join(clips_dir_path, folder)
    shutil.rmtree(folder_path)

# Create a mapping from language to English
lang_to_english = dict(zip(commands_df[language], commands_df['en']))

# Rename folders
for lang_name, english_name in lang_to_english.items():
    try:
        # print(f"Trying to rename: lang_name={lang_name} (type={type(lang_name)}), english_name={english_name} (type={type(english_name)})")
        lang_folder_path = os.path.join(clips_dir_path, lang_name)
        english_folder_path = os.path.join(clips_dir_path, english_name)
        if os.path.exists(lang_folder_path):
            os.rename(lang_folder_path, english_folder_path)
    except Exception as e:
        continue

# Update paths in text files
for text_file in text_files:
    with open(text_file, 'r') as file:
        lines = file.readlines()
    with open(text_file, 'w') as file:
        for line in lines:
            for lang_name, english_name in lang_to_english.items():
                line = line.replace(f'{language}/clips/{lang_name}/', f'{language}/clips/{english_name}/')
            file.write(line)

# --- After renaming folders ---
# Build set of valid English folder names
kept_english_folders = set(lang_to_english[lang] for lang in lang_commands if os.path.exists(os.path.join(clips_dir_path, lang_to_english[lang])))

# Re-process splits CSV and write only valid entries with English folder names
# Clear text files first
for text_file in text_files:
    with open(text_file, 'w') as f:
        f.write("")

with open(splits_csv_path) as csvfile:
    splits_df = pd.read_csv(csvfile)
    for idx, row in tqdm(splits_df.iterrows(), total=len(splits_df), desc='Final txt writing'):
        link = row['LINK']
        lang_folder = link.split('/')[0]
        english_folder = lang_to_english.get(lang_folder)
        if english_folder in kept_english_folders:
            new_link = f"./data_{language}/" + link.replace(lang_folder, english_folder, 1)
            set_type = str(row['SET']).strip().lower()
            if set_type == 'test':
                with open(testing_list, 'a') as f:
                    f.write(f"{new_link}\n")
            elif set_type == 'dev':
                with open(validation_list, 'a') as f:
                    f.write(f"{new_link}\n")
            elif set_type == 'train':
                with open(training_list, 'a') as f:
                    f.write(f"{new_link}\n")

# --- Convert .opus files to .wav and update txt files ---
def convert_opus_to_wav(args):
    folder, file = args
    folder_path = os.path.join(clips_dir_path, folder)
    opus_path = os.path.join(folder_path, file)
    wav_path = opus_path[:-5] + '.wav'
    subprocess.run(['ffmpeg', '-y', '-i', opus_path, wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(opus_path)

# Gather all .opus files
opus_tasks = []
for folder in os.listdir(clips_dir_path):
    folder_path = os.path.join(clips_dir_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.opus'):
                opus_tasks.append((folder, file))

# Convert in parallel (use ProcessPoolExecutor for CPU-bound ffmpeg)
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(convert_opus_to_wav, opus_tasks), total=len(opus_tasks), desc='Converting .opus to .wav'))

# Update txt files to use .wav extension
for text_file in text_files:
    with open(text_file, 'r') as f:
        lines = f.readlines()
    with open(text_file, 'w') as f:
        for line in lines:
            f.write(line.replace('.opus', '.wav'))

# print total number of lines in all three txt files combined
total_lines = 0
for text_file in text_files:
    with open(text_file, 'r') as f:
        total_lines += len(f.readlines())
print(f"Total number of lines across all text files: {total_lines}")

# --- Rename the clips folder itself to data_{language} ---
new_clips_name = f"data_{language}"
parent_dir      = os.path.dirname(clips_dir_path)
new_clips_path  = os.path.join(parent_dir, new_clips_name)

# Make sure the target doesn’t already exist
if os.path.exists(new_clips_path):
    raise FileExistsError(f"{new_clips_path} already exists!")

os.rename(clips_dir_path, new_clips_path)
print(f"Renamed clips folder:\n  {clips_dir_path} →\n  {new_clips_path}")