#!/usr/bin/env python3
import os
import shutil

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    clips_dir = os.path.join('data', 'tr', 'tr', 'data_tr')
    target_dir = os.path.normpath(os.path.join('data', 'turkish-speech-command-dataset', 'data_tr'))

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
    txt_files = ['testing_list.txt', 'validation_list.txt', 'training_list.txt']
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