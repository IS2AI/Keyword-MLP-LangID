#!/usr/bin/env python
import os
import shutil
import json
from audiomentations import AddBackgroundNoise, PolarityInversion
from imutils import paths
import soundfile as sf
import numpy as np
import librosa
import random

# Paths for the RSC dataset and background noise
dataset_path = os.path.join('data', 'ru')         # Original Russian dataset folder
new_dataset_path = os.path.join('Keyword-MLP', 'data_ru')  # Unified dataset destination for Russian data
bn_dataset_path = os.path.join('ESC-50-master', 'audio')      # Background noise folder

# List of commands (keywords)
commands = ["backward", "forward", "right", "left", "down", "up", "go", "stop", 
            "on", "off", "yes", "no", "learn", "follow", "zero", "one", "two", 
            "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", 
            "cat", "dog", "happy", "house", "read", "write", "tree", "visual", "wow"]

# Create new directories for each command if they don't exist
for command in commands:
    command_path = os.path.join(new_dataset_path, command)
    if not os.path.exists(command_path):
        os.makedirs(command_path)

# Predefined augmentation parameters
sample_rate = 16000  # Hz
speeds = [0.8, 1, 1.2]         # time stretch rates
pitches = [-2, 0, 2]           # pitch shifts in semitones
gain_min_factor, gain_max_factor = 0.8, 1.2
noise_percentage_factor = 0.05  # white gaussian noise factor
num_background_noise = 4        # background noise augmentations per file

# Augmentation helper functions
def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    return signal + noise * noise_percentage_factor

def time_stretch(signal, rate):
    return librosa.effects.time_stretch(y=signal, rate=rate)

def pitch_scale(signal, sample_rate, n_steps):
    return librosa.effects.pitch_shift(y=signal, sr=sample_rate, n_steps=n_steps)

def random_gain(signal, min_factor, max_factor):
    gain_rate = random.uniform(min_factor, max_factor)
    return signal * gain_rate

# Prepare background noise augmentation
add_background_noise = AddBackgroundNoise(sounds_path=bn_dataset_path,
                                          min_snr_db=5.0,
                                          max_snr_db=30.0,
                                          noise_transform=PolarityInversion(),
                                          p=1.0)

# Lists to hold new file paths for training and testing
train_list = []
test_list = []

total_train_count = 0
total_test_count = 0

# Process each command folder separately
for command in commands:
    # Folder for the current command in the original dataset
    command_folder = os.path.join(dataset_path, command)
    # List all WAV files in this folder
    wavPaths = list(paths.list_files(command_folder, validExts="wav"))
    wavPaths = sorted(wavPaths)
    
    total_files = len(wavPaths)
    # Compute split index for 57% training, 43% testing (per command)
    train_cutoff = int(total_files * 0.57)
    train_wavs = wavPaths[:train_cutoff]
    test_wavs  = wavPaths[train_cutoff:]
    
    print(f"For command '{command}': total={total_files}, train={len(train_wavs)}, test={len(test_wavs)}")
    
    # --- Process training files with augmentations ---
    for wavPath in train_wavs:
        wav_file = os.path.basename(wavPath)
        subject = os.path.splitext(wav_file)[0]
        # Load the raw audio signal
        raw_signal, _ = librosa.load(wavPath, sr=sample_rate)
        # Destination folder for this command in new_dataset_path
        save_path = os.path.join(new_dataset_path, command)
        
        for speed in speeds:
            ts_signal = time_stretch(raw_signal, speed)
            for pitch in pitches:
                ps_signal = pitch_scale(ts_signal, sample_rate, pitch)
                gs_signal = random_gain(ps_signal, gain_min_factor, gain_max_factor)
                # Prepare file name; for negative pitch, use "n" + abs(pitch)
                pitch_str = f"n{abs(pitch)}" if pitch < 0 else f"{pitch}"
                new_filename = f"{subject}_{speed}_{pitch_str}_0.wav"
                out_path = os.path.join(save_path, new_filename)
                sf.write(out_path, gs_signal, sample_rate)
                train_list.append('./data_ru/' + command + '/' + new_filename)
                total_train_count += 1

                # Augmentation: white noise
                gn_signal = add_white_noise(gs_signal, noise_percentage_factor)
                gn_filename = f"{subject}_{speed}_{pitch_str}_1.wav"
                out_path = os.path.join(save_path, gn_filename)
                sf.write(out_path, gn_signal, sample_rate)
                train_list.append('./data_ru/' + command + '/' + gn_filename)
                total_train_count += 1

                # Augmentation: background noise (multiple copies)
                for j in range(num_background_noise):
                    bn_signal = add_background_noise(gs_signal, sample_rate=sample_rate)
                    bn_filename = f"{subject}_{speed}_{pitch_str}_{j+2}.wav"
                    out_path = os.path.join(save_path, bn_filename)
                    sf.write(out_path, bn_signal, sample_rate)
                    train_list.append('./data_ru/' + command + '/' + bn_filename)
                    total_train_count += 1

    # --- Process testing files with augmentations: Original + 9 augmented versions ---
    for wavPath in test_wavs:
        wav_file = os.path.basename(wavPath)
        subject = os.path.splitext(wav_file)[0]
        raw_signal, _ = librosa.load(wavPath, sr=sample_rate)
        save_path = os.path.join(new_dataset_path, command)
        
        # Save the original file
        out_path = os.path.join(save_path, wav_file)
        sf.write(out_path, raw_signal, sample_rate)
        test_list.append('./data_ru/' + command + '/' + wav_file)
        total_test_count += 1
        
        # Generate 9 augmented versions (for each combination of speeds and pitches)
        for speed in speeds:
            ts_signal = time_stretch(raw_signal, speed)
            for pitch in pitches:
                ps_signal = pitch_scale(ts_signal, sample_rate, pitch)
                gs_signal = random_gain(ps_signal, gain_min_factor, gain_max_factor)
                aug_filename = f"{subject}_{speed}_{pitch}.wav"
                out_path = os.path.join(save_path, aug_filename)
                sf.write(out_path, gs_signal, sample_rate)
                test_list.append('./data_ru/' + command + '/' + aug_filename)
                total_test_count += 1

print("Total training samples (augmented):", total_train_count)
print("Total testing samples:", total_test_count)

# Create label map from command names (alphabetically sorted)
label_list = [label for label in sorted(os.listdir(new_dataset_path))
              if os.path.isdir(os.path.join(new_dataset_path, label)) and label[0] != "_"]
label_map = {idx: label for idx, label in enumerate(label_list)}

# Save the training/testing lists and label map to files
with open(os.path.join(new_dataset_path, "training_list.txt"), "w+") as f:
    f.write("\n".join(train_list))

with open(os.path.join(new_dataset_path, "testing_list.txt"), "w+") as f:
    f.write("\n".join(test_list))

with open(os.path.join(new_dataset_path, "label_map.json"), "w+") as f:
    json.dump(label_map, f)