from audiomentations import AddBackgroundNoise, PolarityInversion
from imutils import paths
import soundfile as sf
import numpy as np
import librosa
import random
import shutil
import json
import os 

# -------------------------
# Setup Paths and Parameters
# -------------------------
# path to the Kazakh Speech Corpus Dataset (KSCD)
dataset_path = os.path.join('data', 'kk')
# path to save the processed dataset
new_dataset_path = os.path.join('Keyword-MLP', 'data_kk')
# path to the background noise dataset
bn_dataset_path = os.path.join('ESC-50-master', 'audio') 

n = 30  # number of randomly selected subjects

commands = ["backward", "forward", "right", "left", "down", "up", "go", "stop", "on", "off", "yes", "no", 
            "learn", "follow", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", 
            "nine", "bed", "bird", "cat", "dog", "happy", "house", "read", "write", "tree", "visual", "wow"]

# Create directories for each command
for command in commands:
    command_path = os.path.join(new_dataset_path, command)
    if not os.path.exists(command_path):
        os.makedirs(command_path)

# -------------------------
# Count files and Split Subjects
# -------------------------
wavPaths = list(paths.list_files(dataset_path, validExts="wav"))
print("Total original wav files:", len(wavPaths))

# Create dictionary: key = subject ID, value = number of samples
dic = {}
for wavPath in wavPaths:
    wav_name = os.path.basename(wavPath)
    sub_id = wav_name.split('.')[0]
    dic[sub_id] = dic.get(sub_id, 0) + 1

print("Total subjects:", len(dic))

# Split subjects into training, validation, and test sets.
train_subjects = []
test_subjects = []
val_subjects = []

for key, val in dic.items():
    if val == 35:
        test_subjects.append(key)
    else:
        train_subjects.append(key)

# Define validation subjects as a subset of test subjects
val_subjects = test_subjects[n:n+10]
train_subjects += test_subjects[n+10:]
test_subjects = test_subjects[:n]

print("# training subjects:", len(train_subjects))
print("# validation subjects:", len(val_subjects))
print("# testing subjects:", len(test_subjects))

# -------------------------
# Augmentation Parameters and Functions
# -------------------------
sample_rate = 16000  # sampling rate in Hz
duration = 1         # audio length in seconds
speeds = [0.8, 1, 1.2]  # time stretch rates
pitches = [-2, 0, 2]    # pitch scales
gain_min_factor, gain_max_factor = 0.8, 1.2  # min-max gain factors
noise_percentage_factor = 0.05  # white gaussian noise factor
num_background_noise = 2  # number of background noises to add

def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    return signal + noise * noise_percentage_factor

def time_stretch(signal, time_stretch_rate):
    return librosa.effects.time_stretch(y=signal, rate=time_stretch_rate)

def pitch_scale(signal, sample_rate, num_semitones):
    return librosa.effects.pitch_shift(y=signal, sr=sample_rate, n_steps=num_semitones)

def random_gain(signal, min_factor, max_factor):
    gain_rate = random.uniform(min_factor, max_factor)
    return signal * gain_rate

def invert_polarity(signal):
    return signal * -1

# Setup background noise augmentation with polarity inversion
add_background_noise = AddBackgroundNoise(sounds_path=bn_dataset_path,
                                          min_snr_db=5.0,
                                          max_snr_db=30.0,
                                          noise_transform=PolarityInversion(),
                                          p=1.0)

# Convert subject lists to sets for faster lookup
train_subjects = set(train_subjects)
val_subjects = set(val_subjects)
test_subjects = set(test_subjects)

# Lists to hold file paths for each split
train_list = []
val_list = []
test_list = []

# -------------------------
# Process Training and Validation Samples
# -------------------------
# For subjects in train or validation sets, we generate several augmentations.
for i, wavPath in enumerate(wavPaths, 1):
    print("[INFO] Processing file: {}/{}".format(i, len(wavPaths)))
    
    # Extract command name and subject ID from the file path
    parts = wavPath.split(os.sep)
    command = parts[-2]
    wav_file = parts[-1]
    subject = wav_file.split('.')[0]
    
    # Load the raw audio signal
    raw_signal, _ = librosa.load(wavPath, sr=sample_rate)
    
    # Define the save path for the command
    save_path = os.path.join(new_dataset_path, command)
    
    if subject in train_subjects or subject in val_subjects:
        # Loop over time stretch rates and pitch scales
        for speed in speeds:
            ts_signal = time_stretch(raw_signal, speed)
            for pitch in pitches:
                # Apply pitch shift and gain augmentation
                ps_signal = pitch_scale(ts_signal, sample_rate, pitch)
                gs_signal = random_gain(ps_signal, gain_min_factor, gain_max_factor)
                
                # Save the baseline augmented file
                wav_file_name = "{}-{}-{}-0.wav".format(subject, speed, pitch)
                sf.write(os.path.join(save_path, wav_file_name), gs_signal, sample_rate)
                
                if subject in train_subjects:
                    train_list.append('./data_kk/' + command + '/' + wav_file_name)
                else:
                    val_list.append('./data_kk/' + command + '/' + wav_file_name)
                
                # Add white gaussian noise
                gn_signal = add_white_noise(gs_signal, noise_percentage_factor)
                gn_wav_file_name = "{}-{}-{}-1.wav".format(subject, speed, pitch)
                sf.write(os.path.join(save_path, gn_wav_file_name), gn_signal, sample_rate)
                
                if subject in train_subjects:
                    train_list.append('./data_kk/' + command + '/' + gn_wav_file_name)
                else:
                    val_list.append('./data_kk/' + command + '/' + gn_wav_file_name)
                    
                # Add random background noise (do this twice)
                for j in range(num_background_noise):
                    bn_signal = add_background_noise(gs_signal, sample_rate=sample_rate)
                    bn_wav_file_name = "{}-{}-{}-{}.wav".format(subject, speed, pitch, j+2)
                    sf.write(os.path.join(save_path, bn_wav_file_name), bn_signal, sample_rate)
                    
                    if subject in train_subjects:
                        train_list.append('./data_kk/' + command + '/' + bn_wav_file_name)
                    else:
                        val_list.append('./data_kk/' + command + '/' + bn_wav_file_name)
    # Skip test subjects in this loop so that we handle them separately

# -------------------------
# Process Test Samples: Original + 9 Augmented Versions
# -------------------------
# Clear out any previously added test files from the first loop.
test_list = []

# We filter only the test subjects
for wavPath in wavPaths:
    parts = wavPath.split(os.sep)
    command = parts[-2]
    wav_file = parts[-1]
    subject = wav_file.split('.')[0]
    
    if subject in test_subjects:
        # Load the raw audio signal
        raw_signal, _ = librosa.load(wavPath, sr=sample_rate)
        save_path = os.path.join(new_dataset_path, command)
        
        # Save the original file (if not already saved)
        sf.write(os.path.join(save_path, wav_file), raw_signal, sample_rate)
        test_list.append('./data_kk/' + command + '/' + wav_file)
        
        # Generate 9 augmented versions (all combinations of speeds and pitches)
        for speed in speeds:
            ts_signal = time_stretch(raw_signal, speed)
            for pitch in pitches:
                # If you want to avoid the (1,0) combination (which is almost identical to the original)
                # uncomment the following lines:
                # if speed == 1 and pitch == 0:
                #     continue
                
                ps_signal = pitch_scale(ts_signal, sample_rate, pitch)
                gs_signal = random_gain(ps_signal, gain_min_factor, gain_max_factor)
                
                aug_wav_file_name = "{}-{}-{}.wav".format(subject, speed, pitch)
                sf.write(os.path.join(save_path, aug_wav_file_name), gs_signal, sample_rate)
                test_list.append('./data_kk/' + command + '/' + aug_wav_file_name)

# -------------------------
# Update List Files and Copy Test Data
# -------------------------
# Write out the updated lists to txt files
with open(os.path.join(new_dataset_path, "training_list.txt"), "w+") as f:
    f.write("\n".join(train_list))

with open(os.path.join(new_dataset_path, "validation_list.txt"), "w+") as f:
    f.write("\n".join(val_list))

with open(os.path.join(new_dataset_path, "testing_list.txt"), "w+") as f:
    f.write("\n".join(test_list))

# Also, update the label map
label_list = [label for label in sorted(os.listdir(new_dataset_path)) 
              if os.path.isdir(os.path.join(new_dataset_path, label)) and label[0] != "_"]
label_map = {idx: label for idx, label in enumerate(label_list)}
with open(os.path.join(new_dataset_path, "label_map.json"), "w+") as f:
    json.dump(label_map, f)

# -------------------------
# Print Final Counts
# -------------------------
print("Train samples:", len(train_list))
print("Validation samples:", len(val_list))
print("Test samples:", len(test_list))
for i, command in enumerate(commands, 1):
    command_dir = os.path.join(new_dataset_path, command)
    wavFiles = list(paths.list_files(command_dir, validExts="wav"))
    print("{}. Command: {}, samples: {}".format(i, command, len(wavFiles)))