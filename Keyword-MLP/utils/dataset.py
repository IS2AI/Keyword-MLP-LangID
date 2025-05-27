import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import functools
import librosa
import glob
import os
from tqdm import tqdm
import multiprocessing as mp
import json

from utils.augment import time_shift, resample, spec_augment
from audiomentations import AddBackgroundNoise


def get_train_val_test_split(root: str, val_file: str, test_file: str):
    """Creates train, val, and test split according to provided val and test files."""
    label_list = [label for label in sorted(os.listdir(root))
                  if os.path.isdir(os.path.join(root, label)) and label[0] != "_"]
    label_map = {idx: label for idx, label in enumerate(label_list)}

    all_files_set = set()
    for label in label_list:
        all_files_set.update(set(glob.glob(os.path.join(root, label, "*.wav"))))

    with open(val_file, "r") as f:
        val_files_set = set(map(lambda a: os.path.join(root, a), f.read().strip().split("\n")))

    with open(test_file, "r") as f:
        test_files_set = set(map(lambda a: os.path.join(root, a), f.read().strip().split("\n")))

    assert len(val_files_set.intersection(test_files_set)) == 0, \
        "Sanity check: No files should overlap between val and test."

    all_files_set -= val_files_set
    all_files_set -= test_files_set

    train_list = list(all_files_set)
    val_list   = list(val_files_set)
    test_list  = list(test_files_set)

    print(f"Number of training samples: {len(train_list)}")
    print(f"Number of validation samples: {len(val_list)}")
    print(f"Number of test samples: {len(test_list)}")

    return train_list, val_list, test_list, label_map


class GoogleSpeechDataset(Dataset):
    """Dataset for multi-task KWS (keyword + language)."""

    def __init__(self,
                 data_list: list,
                 audio_settings: dict,
                 label_map: dict = None,
                 lang_map: dict = None,
                 aug_settings: dict = None,
                 cache: int = 0,
                 default_lang=None):
        super().__init__()

        self.audio_settings = audio_settings
        self.aug_settings   = aug_settings
        self.cache          = cache
        self.default_lang   = default_lang  # Used for single-language datasets

        # If caching is enabled, we load the data into memory (not recommended for inference).
        if cache:
            print("Caching dataset into memory.")
            self.data_list = init_cache(data_list, audio_settings["sr"], cache, audio_settings)
        else:
            self.data_list = data_list

        # If label_map is provided (training), invert it: "0":"backward" → "backward":0
        if label_map is not None:
            self.label_map = {v: int(k) for k, v in label_map.items()}
        else:
            self.label_map = None

        # If lang_map is provided, use it; else default
        if lang_map is not None:
            self.lang_map = lang_map
        else:
            self.lang_map = None

        if aug_settings is not None and "bg_noise" in self.aug_settings:
            self.bg_adder = AddBackgroundNoise(sounds_path=aug_settings["bg_noise"]["bg_folder"])

    def __len__(self):
        return len(self.data_list)

    def extract_language_from_path(self, path):
        """
        Extract language code from file path if possible.
        This handles different dataset formats:
        1. Combined dataset: filename has language prefix like "en_123456.wav"
        2. Single language dataset: derived from path like "data_kk/yes/123456.wav"
        3. Default language if specified and no other language info available
        """
        # First try to get language from filename prefix (combined dataset)
        file_name = os.path.basename(path)
        if '_' in file_name and len(file_name) > 2 and file_name[:2] in self.lang_map:
            return file_name[:2]
        
        # Next try to get language from path (single language dataset)
        path_parts = path.split('/')
        for part in path_parts:
            if part.startswith('data_') and len(part) > 5:
                lang_code = part[5:7]  # Extract 'kk' from 'data_kk'
                if lang_code in self.lang_map:
                    return lang_code
        
        # Use default language if specified
        if self.default_lang is not None:
            return self.default_lang
        
        # If all else fails, return None (will cause error if language is needed)
        return None

    def __getitem__(self, idx):
        # If caching is enabled, self.data_list[idx] will be a tuple (audio, path)
        if isinstance(self.data_list[idx], tuple):
            x, path = self.data_list[idx]
        else:
            path = self.data_list[idx].strip()
            x, _ = librosa.load(path, sr=self.audio_settings["sr"])
        x = self.transform(x)

        # If no label_map provided, we are in inference mode, so return x
        if self.label_map is None:
            return x

        # Extract keyword label from the folder name
        folder_name = path.split("/")[-2]
        kw_label = self.label_map[folder_name]

        # If lang_map is not provided (mono mode), return (x, kw_label)
        if self.lang_map is None:
            return x, kw_label

        # For multitask mode, extract language label
        lang_code = self.extract_language_from_path(path)
        if lang_code is None:
            raise ValueError(f"Could not determine language for file: {path}")
        
        lang_label = self.lang_map[lang_code]
        return x, kw_label, lang_label

    def transform(self, x):
        sr = self.audio_settings["sr"]

        if self.cache < 2:
            if self.aug_settings:
                if "bg_noise" in self.aug_settings:
                    x = self.bg_adder(samples=x, sample_rate=sr)
                if "time_shift" in self.aug_settings:
                    x = time_shift(x, sr, **self.aug_settings["time_shift"])
                if "resample" in self.aug_settings:
                    x, _ = resample(x, sr, **self.aug_settings["resample"])

            x = librosa.util.fix_length(x, size=16000)

            spec = librosa.feature.melspectrogram(y=x, **self.audio_settings)
            x = librosa.feature.mfcc(S=librosa.power_to_db(spec), n_mfcc=self.audio_settings["n_mels"])

        if self.aug_settings and "spec_aug" in self.aug_settings:
            x = spec_augment(x, **self.aug_settings["spec_aug"])

        x = torch.from_numpy(x).float().unsqueeze(0)
        return x


def cache_item_loader(path: str, sr: int, cache_level: int, audio_settings: dict) -> tuple:
    x = librosa.load(path, sr=sr)[0]
    if cache_level == 2:
        x = librosa.util.fix_length(x, size=16000)
        spec = librosa.feature.melspectrogram(y=x, **audio_settings)
        x = librosa.feature.mfcc(S=librosa.power_to_db(spec), n_mfcc=audio_settings["n_mels"])
    return x, path


def init_cache(data_list: list, sr: int, cache_level: int, audio_settings: dict, n_cache_workers: int = 4) -> list:
    """Loads entire dataset into memory for later use."""
    cache = []
    loader_fn = functools.partial(cache_item_loader,
                                  sr=sr,
                                  cache_level=cache_level,
                                  audio_settings=audio_settings)

    pool = mp.Pool(n_cache_workers)
    for audio in tqdm(pool.imap(func=loader_fn, iterable=data_list), total=len(data_list)):
        cache.append(audio)
    pool.close()
    pool.join()

    return cache


def get_loader(data_list, config, train=True, default_lang=None):
    """
    Creates dataloaders for training or validation/test.
    - If train=True, we load label_map & lang_map from config so the dataset yields (x, kw_label, lang_label).
    - If train=False (in typical eval mode), also yields (x, kw_label, lang_label).
      But for pure inference, you might use a separate script that sets label_map=None, lang_map=None.
    
    Parameters:
        default_lang: Two-letter language code to use if the dataset doesn't have language prefixes.
                     This is useful for evaluating single-language datasets with multitask models.
    """
    # Detect dataset type from the path patterns
    is_single_lang_dataset = False
    if len(data_list) > 0:
        sample_path = data_list[0]
        if 'data_kk/' in sample_path or 'data_ru/' in sample_path or 'data_tt/' in sample_path:
            is_single_lang_dataset = True
            # Extract language code from path
            for lang_code in ['kk', 'ru', 'tt', 'ar', 'tr', 'en']:
                if f'data_{lang_code}/' in sample_path:
                    default_lang = lang_code
                    break

    with open(config["label_map"], "r") as f:
        label_map = json.load(f)

    model_cfg = config["hparams"]["model"]
    if model_cfg.get("multitask", False) or model_cfg.get("lang_only", False):
        with open(config["lang_map"], "r") as f2:
            lang_map = json.load(f2)
    else:
        lang_map = None

    dataset = GoogleSpeechDataset(
        data_list=data_list,
        label_map=label_map,
        lang_map=lang_map,
        audio_settings=config["hparams"]["audio"],
        aug_settings=config["hparams"]["augment"] if train else None,
        cache=config["exp"]["cache"],
        default_lang=default_lang
    )

    # Get appropriate worker_init_fn and generator if available in config
    worker_init_fn = config["hparams"].get("dataloader", {}).get("worker_init_fn", None)
    generator = config["hparams"].get("dataloader", {}).get("generator", None)

    loader = DataLoader(
        dataset,
        batch_size=config["hparams"]["batch_size"],
        num_workers=config["exp"]["n_workers"],
        pin_memory=config["exp"]["pin_memory"],
        shuffle=True if train else False,
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    return loader