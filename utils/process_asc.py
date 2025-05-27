import argparse
import shutil
import os
from pathlib import Path
import pandas as pd

# Commands to keep
ALLOWED_COMMANDS = {
    "backward", "four", "right", "three", "yes",
    "down",     "left",  "seven", "zero",
    "eight",    "nine",  "six",   "two",
    "five",     "no",    "stop",  "up",
    "forward",  "one"
}


def find_raw_dir(base: Path, subdir: str) -> Path:
    """
    Descend into nested subfolders until one contains ALLOWED_COMMANDS.
    """
    candidate = base / subdir
    if not candidate.is_dir():
        raise RuntimeError(f"Raw subdir not found: {candidate}")
    while True:
        children = [d for d in candidate.iterdir() if d.is_dir()]
        if any(d.name in ALLOWED_COMMANDS for d in children):
            return candidate
        if len(children) == 1:
            candidate = children[0]
            continue
        return candidate


def rename_raw_dir(raw_dir: Path, new_name: str) -> Path:
    """
    Rename raw_dir to raw_dir.parent/new_name. Return new path.
    """
    target = raw_dir.parent / new_name
    if raw_dir.resolve() == target.resolve():
        return raw_dir
    if target.exists():
        raise RuntimeError(f"Target directory already exists: {target}")
    raw_dir.rename(target)
    print(f"Renamed {raw_dir} → {target}")
    return target


def clean_commands(data_ar: Path):
    """Delete subdirectories not in ALLOWED_COMMANDS."""
    for d in data_ar.iterdir():
        if d.is_dir() and d.name not in ALLOWED_COMMANDS:
            print(f"Removing folder: {d}")
            shutil.rmtree(d)


def csvs_to_lists(data_root: Path, data_ar: Path):
    """
    Read train.csv & val.csv, filter on ALLOWED_COMMANDS,
    write training_list.txt & validation_list.txt into data_ar.
    Each output ends with a blank line.
    """
    mapping = {
        "train.csv": "training_list.txt",
        "val.csv":   "validation_list.txt",
    }
    for csv_name, out_name in mapping.items():
        csv_p = data_root / csv_name
        if not csv_p.exists():
            print(f"[!] {csv_p} missing, skipping.")
            continue
        df = pd.read_csv(csv_p, usecols=["file", "class"] )
        df = df[df["class"].isin(ALLOWED_COMMANDS)]
        lines = []
        for _, row in df.iterrows():
            rel = row["file"]
            if rel.startswith("dataset/"):
                rel = rel[len("dataset/"):]
            lines.append(f"./data_ar/{rel}")
        out_p = data_ar / out_name
        # write with trailing blank line
        out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  → {out_name}: {len(lines)} lines")


def load_list(path: Path) -> set:
    """Return set of paths without leading './'."""
    s = set()
    for L in path.read_text(encoding="utf-8").splitlines():
        rel = L.strip().split(",", 1)[0]
        if rel.startswith("./"):
            rel = rel[2:]
        s.add(rel)
    return s


def build_testing_list(data_ar: Path, train_set: set, val_set: set):
    """
    Walk data_ar for .wav, exclude train/val, write testing_list.txt in data_ar.
    The output will have a trailing blank line.
    """
    lines = []
    for dp, _, fns in os.walk(data_ar):
        for fn in fns:
            if not fn.lower().endswith('.wav'):
                continue
            full = Path(dp) / fn
            rel = full.relative_to(data_ar).as_posix()
            key = f"data_ar/{rel}"
            if key in train_set or key in val_set:
                continue
            lines.append(f"./{key}")
    out_p = data_ar / 'testing_list.txt'
    out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  → Wrote testing_list.txt: {len(lines)} lines")


def main():
    parser = argparse.ArgumentParser(
        description="Process ASC and generate lists inside data_ar"
    )
    parser.add_argument(
        '--data_root', type=Path,
        default=Path('data/arabic-speech-commands-dataset'),
        help='where train.csv & val.csv live'
    )
    parser.add_argument(
        '--raw_subdir', type=str,
        default='dataset',
        help='folder under data_root containing command subdirs'
    )
    args = parser.parse_args()

    # 1) find nested raw folder
    raw_dir = find_raw_dir(args.data_root, args.raw_subdir)
    print(f"Found raw folder: {raw_dir}")

    # 2) rename it to data_ar
    data_ar = rename_raw_dir(raw_dir, 'data_ar')

    # 3) clean commands
    print("\nCleaning commands…")
    clean_commands(data_ar)

    # 4) CSV → training/validation lists
    print("\nGenerating CSV-based lists…")
    csvs_to_lists(args.data_root, data_ar)

    # 5) build testing_list from leftover WAVs
    print("\nBuilding testing_list.txt…")
    train_set = load_list(data_ar / 'training_list.txt')
    val_set   = load_list(data_ar / 'validation_list.txt')
    build_testing_list(data_ar, train_set, val_set)

if __name__ == '__main__':
    main()