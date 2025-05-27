#!/usr/bin/env python3
import os
import shutil

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    clips_dir = os.path.join('data', 'ar', 'ar', 'data_ar')
    target_dir = os.path.normpath(os.path.join('data', 'arabic-speech-commands-dataset', 'dataset', 'data_ar'))

    # Delete folders 'left' and 'down' due to mismatch of translations
    for folder in ['left', 'down']:
        path = os.path.join(clips_dir, folder)
        if os.path.isdir(path):
            shutil.rmtree(path)

    # Update txt files to remove references to 'left' and 'down'
    txt_files = ['testing_list.txt', 'validation_list.txt', 'training_list.txt']
    for txt in txt_files:
        path = os.path.join(clips_dir, txt)
        if os.path.isfile(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            lines = [l for l in lines if '/left/' not in l and '/down/' not in l]
            with open(path, 'w') as f:
                f.writelines(lines)

    # Merge audio files/folders
    for name in os.listdir(clips_dir):
        src_path = os.path.join(clips_dir, name)
        dest_path = os.path.join(target_dir, name)
        if os.path.isdir(src_path):
            if os.path.isdir(dest_path):
                for fname in os.listdir(src_path):
                    src_file = os.path.join(src_path, fname)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, os.path.join(dest_path, fname))
            else:
                shutil.copytree(src_path, dest_path)

    # Append txt files to target directory
    for txt in txt_files:
        src_path = os.path.join(clips_dir, txt)
        dest_path = os.path.join(target_dir, txt)
        if os.path.isfile(src_path):
            with open(src_path, 'r') as f:
                lines = f.readlines()
            mode = 'a' if os.path.isfile(dest_path) else 'w'
            with open(dest_path, mode) as f:
                f.writelines(lines)

if __name__ == '__main__':
    main()