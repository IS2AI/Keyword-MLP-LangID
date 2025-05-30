# Understanding the Config File

The config file contains all the hyperparameters and various settings regarding your training runs. I'll break down the numerous settings here as clearly as possible.

## Dataset Paths

```
data_root: ./data/                           # Where you have extracted the google speech commands v2 dataset.
train_list_file: ./data/training_list.txt    # Contains paths to your training .wav files.
val_list_file: ./data/validation_list.txt    # Contains paths to your validation .wav files.
test_list_file: ./data/testing_list.txt      # Contains paths to your test .wav files.
label_map: ./data/label_map.json             # A json file containing {id: label} key value pairs.
lang_map: ./data/lang_map.json               # A json file containing {language: id} key value pairs. Used in multilingual settings.
```

## Experiment Settings

```
exp:
    wandb: False                           # Whether to use wandb or not
    wandb_api_key: <path/to/api/key>       # Path to your key. Ignored if wandb is False. If blank, looks for key in the ${WANDB_API_KEY} env variable.
    proj_name: torch-kw-mlp                # Name of your wandb project. Ignored if wandb is False.
    exp_dir: ./runs                        # Your checkpoints will be saved locally at exp_dir/exp_name
    exp_name: kw-mlp-0.1.0                 # ..for example, ./runs/kw-mlp-0.1.0/something.pth
    device: auto                           # "auto" checks whether cuda is available; if not, uses cpu. You can also put in "cpu" or "cuda" as device.
                                           # only single   device training is supported currently.
    log_freq: 20                           # Saves logs every log_freq steps
    log_to_file: True                      # Saves logs to exp_dir/exp_name/training_logs.txt
    log_to_stdout: True                    # Prints logs to stdout
    val_freq: 1                            # Validate every val_freq epochs
    n_workers: 2                           # Number of workers for dataloader --- best to set to number of CPUs on machine
    pin_memory: True                       # Pin memory argument for dataloader
    cache: 2                               # 0 -> no cache | 1 -> cache wav arrays | 2 -> cache MFCCs (and also prevents wav augmentations like time_shift,
                                           # resampling and add_background_noise)
```

## Hyperparameters
```
hparams:                    # everything nested under hparams are hyperparamters, and will be logged as wandb hparams as well.
    ...
    ...
```
### Basic settings
```
hparams:
    restore_ckpt:            # Path to ckpt, if resuming an interrupted training run. Ckpt must have optimizer state as well.
    seed: 0                  # Random seed for determinism
    batch_size: 256          # Batch size
    start_epoch: 0           # Start epoch, 0 by default.   
    n_epochs: 140            # How many epochs will be trained. (1 epoch = (len(dataset) / batch_size) steps)
    l_smooth: 0.1            # If a positive float, uses LabelSmoothingLoss instead of the vanilla CrossEntropyLoss
```

### Audio Processing
```
hparams:
    ...
    ...
    audio:
        sr: 16000            # sampling rate
        n_mels: 40           # number of mel bands for melspectrogram (and MFCC)
        n_fft: 480           # n_fft, window length, hop length, center are also all args for calculating the melspectrogram
        win_length: 480      # Check the docs here for further explanation: 
        hop_length: 160      # https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram
        center: False        # MFCC conversion is currently done on CPU with librosa. May add in a CUDA MFCC conversion later (with nnAudio)
```

### Model Settings
```
hparams:
    ...
    ...
    model:
        type: kw-mlp         # Selects the KW-MLP architecture
        input_res: [40, 98]  # Shape of input spectrogram (n_mels x T)
        patch_res: [40, 1]   # Resolution of patches
        num_classes: 35      # Number of classes
        num_langs: 15        # Number of languages, used in multilingual settings.
        channels: 1          # MFCCs are single channel inputs
        dim: 64              # Patch embedding dim (d)
        depth: 12            # Number of gated MLP blocks (L)
        pre_norm: False      # Prenorm or Postnorm gated-MLP. PostNorm has been shown to perform better
        prob_survival: 0.9   # Each gated MLP block has a 0.1 probability of being dropped, as a regularization scheme
        dropout: 0.2         # Dropout rate for regularization.
        multitask: True      # Whether to use multitask learning.
        lang_only: True      # If true, the model is trained only for language classification.
```

### Optimizer & Scheduling
```
hparams:
    ...
    ...
    optimizer:               # AdamW with an lr of 0.001 and weight decay of 0.1, as in the paper.
        opt_type: adamw      # Please modify get_optimizer() in utils/opt.py if you want to add support for more optimizer variants.
        opt_kwargs:
          lr: 0.001
          weight_decay: 0.1
    
    scheduler:               # Warmup scheduling for 10 epochs and cosine annealing, as in the paper.
        n_warmup: 10         # Please modify get_scheduler() in utils/scheduler.py if you want to add support for other scheduling techniques.
        max_epochs: 140      # Up to which epoch the normal scheduler will be run.
        scheduler_type: cosine_annealing
```

### Augmentation
```
hparams:
    ...
    ...
    augment:                 # Augmentations are applied only during training. In the paper, only spec_aug is used. Resample, time_shift and
                             # bg_noise are available, like in the Keyword-Transformer paper, but increases training time significantly.
                             # Make sure to comment out resample, time_shift and bg_noise if the goal is to reproduce the results of KW-MLP. 

        # resample:          # Randomly resamples between 85% and 115%
            # r_min: 0.85
            # r_max: 1.15
        
        # time_shift:        # Randomly shifts samples left or right, up to 10%
            # s_min: -0.1
            # s_max: 0.1

        # bg_noise:          # Adds background noise from a folder containing noise files. Make sure folder only contains .wav files
            # bg_folder: ./data/_background_noise_/

        spec_aug:            # Spectral augmentation. SpecAug is applied on the CPU currently, with a Numba JIT compiled function. May provide a 
                             # CUDA SpecAug later.
            n_time_masks: 2
            time_mask_width: 25
            n_freq_masks: 2
            freq_mask_width: 7
```