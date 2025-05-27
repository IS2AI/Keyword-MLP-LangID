#!/usr/bin/env python3
import json
import os
import glob
from argparse import ArgumentParser
from utils.dataset import get_train_val_test_split

def replace_labels_in_list(file_list, replacements):
    """Replace occurrences in a list of paths based on a replacement dictionary."""
    updated_list = []
    for path in file_list:
        for old, new in replacements.items():
            path = path.replace(f"/{old}/", f"/{new}/")
        updated_list.append(path)
    return updated_list


def main(args):
    # Define replacement map for labels and folders
    replacements = {"marvin": "read", "sheila": "write"}

    # Step 1: Load train/val/test splits and original label map
    train_list, val_list, test_list, label_map = get_train_val_test_split(
        args.data_root, args.val_list_file, args.test_list_file
    )

    # Step 2: Rewrite list entries and update label_map indices
    train_list = replace_labels_in_list(train_list, replacements)
    val_list   = replace_labels_in_list(val_list, replacements)
    test_list  = replace_labels_in_list(test_list, replacements)

    for idx, label in label_map.items():
        if label in replacements:
            label_map[idx] = replacements[label]

    # Step 3: Save updated lists and label map
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "training_list.txt"), "w") as f:
        f.write("\n".join(train_list))

    with open(os.path.join(args.out_dir, "validation_list.txt"), "w") as f:
        f.write("\n".join(val_list))

    with open(os.path.join(args.out_dir, "testing_list.txt"), "w") as f:
        f.write("\n".join(test_list))

    with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # Step 4: Rename folders on disk after lists are saved
    for old_name, new_name in replacements.items():
        old_path = os.path.join(args.data_root, old_name)
        new_path = os.path.join(args.data_root, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed folder: {old_name} -> {new_name}")
        else:
            print(f"Folder '{old_name}' not found. Skipping rename.")

    print("Saved updated data lists, label map, and renamed folders.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--val_list_file", type=str, required=True,
        help="Path to validation_list.txt."
    )
    parser.add_argument(
        "-t", "--test_list_file", type=str, required=True,
        help="Path to testing_list.txt."
    )
    parser.add_argument(
        "-d", "--data_root", type=str, required=True,
        help="Root directory of speech commands v2 dataset."
    )
    parser.add_argument(
        "-o", "--out_dir", type=str, required=True,
        help="Output directory for updated lists and label map."
    )
    args = parser.parse_args()

    main(args)