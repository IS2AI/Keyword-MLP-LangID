import os
import random
import soundfile as sf
import numpy as np
import librosa
from audiomentations import AddBackgroundNoise, PolarityInversion
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

# --------------------- Global deterministic setup ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

global_rng = random.Random(SEED)  # used only for sampling in process_txt

# --------------------- Augmentation parameters --------------------------------
sample_rate = 16_000
speeds = [0.8, 1.0, 1.2]
pitches = [-2, 0, 2]
gain_min, gain_max = 0.8, 1.2
noise_pct = 0.05
num_bg = 2  # two background‑noise variations

# Targets chosen so that 26 commands → ≈ 71 k / 10 k / 10 k per split
target_per_cmd = {
    "training_list.txt": 3_300,
    "validation_list.txt": 385,
    "testing_list.txt":   385,
}

# --------------------- Background‑noise augmenter -----------------------------

def create_bg_augmenter():
    path = os.path.join("ESC-50-master", "audio")
    return AddBackgroundNoise(
        sounds_path=path,
        min_snr_db=5.0,
        max_snr_db=30.0,
        noise_transform=PolarityInversion(),
        p=1.0,
    )

bg_augment = create_bg_augmenter()

# --------------------- Single‑file augmentation -------------------------------

def augment_and_save(args):
    file_path, cmd, filename, data_dir, task_idx = args

    # per‑task deterministic RNGs
    r = random.Random(SEED + task_idx)
    np_r = np.random.RandomState(SEED + task_idx)

    try:
        sig, sr = sf.read(file_path)
    except Exception as e:
        print(f"WARNING: cannot read {file_path}: {e}")
        return []

    if sr != sample_rate:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=sample_rate)

    prefix = os.path.splitext(filename)[0]
    save_dir = os.path.join(data_dir, cmd)
    os.makedirs(save_dir, exist_ok=True)

    out = []
    for speed in speeds:
        stretched = np.interp(
            np.linspace(0, len(sig), int(len(sig) / speed)),
            np.arange(len(sig)), sig,
        )
        for pitch in pitches:
            shifted = librosa.effects.pitch_shift(stretched, sr=sample_rate, n_steps=pitch)
            gain = r.uniform(gain_min, gain_max)
            base = shifted * gain

            # 0) gain‑scaled
            fn0 = f"{prefix}-{speed}-{pitch}-0.wav"
            sf.write(os.path.join(save_dir, fn0), base, sample_rate)
            out.append(fn0)

            # 1) white‑noise
            noise = base + np_r.normal(0, base.std(), base.shape) * noise_pct
            fn1 = f"{prefix}-{speed}-{pitch}-1.wav"
            sf.write(os.path.join(save_dir, fn1), noise, sample_rate)
            out.append(fn1)

            # 2,3) background noise
            for j in range(num_bg):
                bn = bg_augment(base, sample_rate=sample_rate)
                fnj = f"{prefix}-{speed}-{pitch}-{j+2}.wav"
                sf.write(os.path.join(save_dir, fnj), bn, sample_rate)
                out.append(fnj)

    base_dir = os.path.basename(data_dir)  # "data_ar"
    return [f"./{base_dir}/{cmd}/{fn}" for fn in out]

# --------------------- Split processing ---------------------------------------

def process_split(txt_name: str, data_dir: str):
    txt_path = os.path.join(data_dir, txt_name)
    with open(txt_path) as f:
        originals = [ln.strip() for ln in f if ln.strip()]

    # Deduplicate & normalise
    originals = sorted({ln if ln.startswith("./") else f"./{ln}" for ln in originals})

    # Build tasks
    tasks = []
    for idx, rel in enumerate(originals):
        rel_clean = rel[2:]  # drop "./"
        _, cmd, wav = rel_clean.split("/")
        wav_path = os.path.join(data_dir, cmd, wav)
        tasks.append((wav_path, cmd, wav, data_dir, idx))

    # Parallel augmentation (deterministic because task indices are stable)
    with ProcessPoolExecutor() as pool:
        aug_lists = list(pool.map(augment_and_save, tasks))
    all_aug = [p for sub in aug_lists for p in sub]

    # Group by command
    orig_by_cmd, aug_by_cmd = defaultdict(list), defaultdict(list)
    for rel in originals:
        orig_by_cmd[rel.split("/")[2]].append(rel)
    for aug in all_aug:
        aug_by_cmd[aug.split("/")[2]].append(aug)

    target = target_per_cmd[txt_name]
    final = []
    for cmd in sorted(orig_by_cmd):
        raw = orig_by_cmd[cmd]
        augs = aug_by_cmd.get(cmd, [])

        if len(raw) >= target:
            # Down‑sample raw to target to keep uniformity (but still deterministic)
            chosen_raw = global_rng.sample(raw, target)
            final.extend(chosen_raw)
            continue

        # Always take all raw, then add augmentations / repeats to hit target
        needed = target - len(raw)
        selected = list(raw)
        if augs:
            if len(augs) >= needed:
                selected += global_rng.sample(augs, needed)
            else:
                reps = needed // len(augs) + 1
                selected += global_rng.sample(augs * reps, needed)
        else:
            # No augmentations produced (corrupt audio etc.). Duplicate raw deterministically
            reps = needed // len(raw) + 1
            selected += global_rng.sample(raw * reps, needed)
        final.extend(selected)

    final = sorted(final)

    # Backup & overwrite
    os.rename(txt_path, txt_path + ".backup")
    with open(txt_path, "w") as f:
        f.write("\n".join(final) + "\n")

    print(f"{txt_name}: {len(final)} total (target {target} × {len(orig_by_cmd)} = {target*len(orig_by_cmd)})")

# --------------------- Clean‑up helper ----------------------------------------

def cleanup_unused(data_dir: str, split_files):
    """Remove .wav files that are not referenced in any split list."""
    referenced = set()
    for txt in split_files:
        with open(os.path.join(data_dir, txt)) as f:
            for ln in f:
                rel = ln.strip()
                if rel.startswith("./"):
                    rel = rel[2:]
                parts = rel.split("/", 2)
                if len(parts) == 3:
                    referenced.add(os.path.abspath(os.path.join(data_dir, parts[1], parts[2])))
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn.endswith(".wav"):
                full = os.path.abspath(os.path.join(root, fn))
                if full not in referenced:
                    try:
                        os.remove(full)
                    except OSError:
                        pass

# --------------------- Entrypoint --------------------------------------------

if __name__ == "__main__":
    DATA_DIR = os.path.join("data", "arabic-speech-commands-dataset", "dataset", "data_ar")
    SPLITS = ["training_list.txt", "validation_list.txt", "testing_list.txt"]

    for split in SPLITS:
        process_split(split, DATA_DIR)

    cleanup_unused(DATA_DIR, SPLITS)