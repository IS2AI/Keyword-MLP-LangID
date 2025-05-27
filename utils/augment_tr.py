import os
import random
import soundfile as sf
import numpy as np
import librosa
from audiomentations import AddBackgroundNoise, PolarityInversion
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

# -------------------- Global deterministic setup -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
GLOBAL_RNG = random.Random(SEED)  # used for sampling inside process_txt

# -------------------- Augmentation parameters --------------------------------
SAMPLE_RATE = 16_000
SPEEDS = [0.8, 1.0, 1.2]
PITCHES = [-2, 0, 2]
GAIN_MIN, GAIN_MAX = 0.8, 1.2
WHITE_NOISE_PCT = 0.05
NUM_BG = 2  # two background‑noise variations per speed/pitch

TARGETS = {
    "training_list.txt": 3_300,
    "validation_list.txt": 385,
    "testing_list.txt":   385,
}

# -------------------- Background‑noise augmenter -----------------------------

def create_bg_augmenter():
    path = os.path.join("ESC-50-master", "audio")
    return AddBackgroundNoise(
        sounds_path=path,
        min_snr_db=5.0,
        max_snr_db=30.0,
        noise_transform=PolarityInversion(),
        p=1.0,
    )

BG_AUGMENT = create_bg_augmenter()

# -------------------- Single‑file augmentation -------------------------------

def augment_and_save(args):
    """Return list of relative paths for 36 deterministic augmentations."""
    file_path, cmd, filename, data_dir, task_idx = args
    rng = random.Random(SEED + task_idx)
    np_rng = np.random.RandomState(SEED + task_idx)

    try:
        sig, sr = sf.read(file_path)
    except Exception as e:
        print(f"[WARN] Skipping corrupt {file_path}: {e}")
        return []

    if sr != SAMPLE_RATE:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=SAMPLE_RATE)

    prefix = os.path.splitext(filename)[0]
    save_dir = os.path.join(data_dir, cmd)
    os.makedirs(save_dir, exist_ok=True)

    out = []
    for speed in SPEEDS:
        stretched = np.interp(
            np.linspace(0, len(sig), int(len(sig) / speed)),
            np.arange(len(sig)), sig,
        )
        for pitch in PITCHES:
            shifted = librosa.effects.pitch_shift(stretched, sr=SAMPLE_RATE, n_steps=pitch)
            gain = rng.uniform(GAIN_MIN, GAIN_MAX)
            base = shifted * gain

            # 0) gain‑scaled
            fn0 = f"{prefix}-{speed}-{pitch}-0.wav"
            sf.write(os.path.join(save_dir, fn0), base, SAMPLE_RATE)
            out.append(fn0)

            # 1) white‑noise
            wn = base + np_rng.normal(0, base.std(), base.shape) * WHITE_NOISE_PCT
            fn1 = f"{prefix}-{speed}-{pitch}-1.wav"
            sf.write(os.path.join(save_dir, fn1), wn, SAMPLE_RATE)
            out.append(fn1)

            # 2,3) background noise
            for j in range(NUM_BG):
                bn = BG_AUGMENT(base, sample_rate=SAMPLE_RATE)
                fnj = f"{prefix}-{speed}-{pitch}-{j+2}.wav"
                sf.write(os.path.join(save_dir, fnj), bn, SAMPLE_RATE)
                out.append(fnj)

    base_dir = os.path.basename(data_dir)  # "data_tr"
    return [f"./{base_dir}/{cmd}/{fn}" for fn in out]

# -------------------- Split processing ---------------------------------------

def process_split(split_name: str, data_dir: str):
    txt_path = os.path.join(data_dir, split_name)
    with open(txt_path) as f:
        originals = [ln.strip() for ln in f if ln.strip()]

    originals = sorted({ln if ln.startswith("./") else f"./{ln}" for ln in originals})

    # Build deterministic task list
    tasks = []
    for idx, rel in enumerate(originals):
        clean = rel[2:]
        _, cmd, wav = clean.split("/")
        wav_path = os.path.join(data_dir, cmd, wav)
        tasks.append((wav_path, cmd, wav, data_dir, idx))

    # Augment in parallel
    with ProcessPoolExecutor() as pool:
        aug_lists = list(pool.map(augment_and_save, tasks))
    all_aug = [p for sub in aug_lists for p in sub]

    # Group by command
    orig_by_cmd, aug_by_cmd = defaultdict(list), defaultdict(list)
    for rel in originals:
        orig_by_cmd[rel.split("/")[2]].append(rel)
    for aug in all_aug:
        aug_by_cmd[aug.split("/")[2]].append(aug)

    target = TARGETS[split_name]
    final = []
    for cmd in sorted(orig_by_cmd):
        raws = orig_by_cmd[cmd]
        augs = aug_by_cmd.get(cmd, [])
        selected = list(raws)  # keep ALL raw samples

        if len(selected) < target:  # need more samples -> add augmentations
            needed = target - len(selected)
            if len(augs) >= needed:
                selected += GLOBAL_RNG.sample(augs, needed)
            else:
                reps = needed // len(augs) + 1
                selected += GLOBAL_RNG.sample(augs * reps, needed)
        # If raw already >= target, **do not** drop anything and **do not** add more.
        final.extend(selected)

    final = sorted(final)
    os.rename(txt_path, txt_path + ".backup")
    with open(txt_path, "w") as f:
        f.write("\n".join(final) + "\n")

    print(f"{split_name}: {len(final)} lines written; targets {target} (min), raw never removed.")

# -------------------- Cleanup unused augmentations ---------------------------

def cleanup_unused(data_dir: str, split_files):
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
            if not fn.endswith(".wav"):
                continue
            full = os.path.abspath(os.path.join(root, fn))
            if full not in referenced:
                try:
                    os.remove(full)
                except OSError:
                    pass

# -------------------- Entrypoint --------------------------------------------
if __name__ == "__main__":
    DATA_DIR = os.path.join("data", "turkish-speech-command-dataset", "data_tr")
    SPLITS = ["training_list.txt", "validation_list.txt", "testing_list.txt"]

    for split in SPLITS:
        process_split(split, DATA_DIR)

    cleanup_unused(DATA_DIR, SPLITS)
