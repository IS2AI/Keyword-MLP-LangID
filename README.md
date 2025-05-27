# Keyword-MLP-LangID

## Overview

Keyword-MLP-LangID is a project designed to train and evaluate models for multilingual keyword spotting and language identification. This project addresses the challenges of voice interaction in multilingual environments by proposing a unified multitask model that performs both Speech Command Recognition (SCR) and Language Identification (LID) simultaneously.

## Final Models

The final models are available on Hugging Face:
- [Multi-LID Model](https://huggingface.co/artur-muratov/kw-mlp-multi-lid)
- [Multi-SCR-LID Model](https://huggingface.co/artur-muratov/kw-mlp-multi-scr-lid)
- [Mono Models](https://huggingface.co/artur-muratov/kw-mlp-mono-kk) (only language code differs)

## Multilingual Dataset

The multilingual speech commands dataset is available on Hugging Face:
- [Multilingual Speech Commands Dataset](https://huggingface.co/datasets/artur-muratov/multilingual-speech-commands-15lang)

## Getting Started

### Prerequisites

- Python 3.x
- Additional Python packages as specified in `Keyword-MLP/requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/IS2AI/Keyword-MLP-LangID
   cd Keyword-MLP-LangID
   ```

2. Install the required packages:
   ```bash
   pip install -r Keyword-MLP/requirements.txt
   ```

### Using the Final Dataset

1. Clone the multilingual speech commands dataset:
   ```bash
   cd Keyword-MLP
   git clone https://huggingface.co/datasets/artur-muratov/multilingual-speech-commands-15lang
   mv multilingual-speech-commands-15lang data_all
   ```

2. Train the model (example):
   ```bash
   python train.py --conf configs/config_mul_lc_final.yaml
   ```
   - `--conf`: Specifies the configuration file to use for training. This file contains all the necessary settings and hyperparameters for the training process.

3. Evaluate the model (example):
   ```bash
   python eval.py --conf configs/config_mul_lc_final.yaml --ckpt runs/kw-mlp-mul-final-2048-dropout-0.2/best.pth --test_list data_all/testing_list.txt
   ```
   - `--conf`: Specifies the configuration file to use for evaluation.
   - `--ckpt`: Path to the checkpoint file containing the trained model weights.
   - `--test_list`: File containing the list of test data.
   - `--out_fig` (optional): Path where the output confusion matrix image will be saved.

### Configuration

The configuration files are detailed in `Keyword-MLP/docs/config_file_explained.md`.

### Alternatively, Constructing the Final Dataset

The `utils` folder provides all the necessary code to construct the final dataset:

1. Download the kk/ru/tt dataset:
   ```bash
   git clone https://huggingface.co/datasets/artur-muratov/multilingual-speech-commands-3lang-raw
   mv multilingual-speech-commands-3lang-raw data
   ```

   The dataset is structured into three language folders:
   - `kk` for Kazakh
   - `ru` for Russian
   - `tt` for Tatar

2. For MSWC dataset:
   - **`download_mswc.py`**: Downloads audio and split files for specified languages from the Multilingual Spoken Words Corpus (MSWC).
     ```bash
     python utils/download_mswc.py
     ```
   - **`run_mswc_process_batch.py`**: Automates processing of MSWC data for multiple languages.
     ```bash
     python utils/run_mswc_process_batch.py
     ```

3. For Arabic and Turkish datasets:
   - **`download_asc.py`**: Downloads the Arabic Speech Commands dataset from Kaggle.
     ```bash
     python utils/download_asc.py
     ```
   - **`download_trsc.py`**: Downloads the Turkish Speech Command dataset from Kaggle.
     ```bash
     python utils/download_trsc.py
     ```
   - **`process_asc.py`**: Processes the Arabic Speech Commands dataset.
     ```bash
     python utils/process_asc.py
     ```
   - **`process_trsc.py`**: Processes the Turkish Speech Command dataset.
     ```bash
     python utils/process_trsc.py
     ```
   - **`merge_ar.py`**: Merges Arabic dataset directories and updates text files.
     ```bash
     python utils/merge_ar.py
     ```
   - **`merge_tr.py`**: Merges Turkish dataset directories and updates text files.
     ```bash
     python utils/merge_tr.py
     ```
   - **`augment_ar.py`**: Augments the Arabic dataset with various audio transformations.
     ```bash
     python utils/augment_ar.py
     ```
   - **`augment_tr.py`**: Augments the Turkish dataset with various audio transformations.
     ```bash
     python utils/augment_tr.py
     ```

4. Augment after all merges and finished MSWC dataset:
   ```bash
   python utils/aug_mswc.py
   ```

5. Download the Google Speech Commands V2 dataset:
   ```bash
   cd Keyword-MLP
   sh ./download_gspeech_v2.sh data
   python make_data_list.py -v data/validation_list.txt -t data/testing_list.txt -d ./data -o ./data
   cd ..
   ```

6. Finally, combine all datasets:
   ```bash
   python utils/combine_all.py
   ```

### Datasets

This project utilizes several datasets for training and evaluation. Below are the citations for each dataset used:

- **Google Speech Commands V2**: 
  - Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition. ArXiv e-prints. Available: https://arxiv.org/abs/1804.03209

- **Multilingual Spoken Words Corpus (MSWC)**: 
  - Mazumder, M., Chitlangia, S., Banbury, C., Kang, Y., Ciro, J. M., Achorn, K., Galvez, D., Sabini, M., Mattson, P., Kanter, D., Diamos, G., Warden, P., Meyer, J., & Reddi, V. J. (2021). Multilingual spoken words corpus. In Proc. of the NeurIPS Datasets and Benchmarks Track. Available: https://mlcommons.org/en/multilingual-spoken-words

- **Arabic Speech Commands Dataset**: 
  - Ghandoura, A., Hjabo, F., & Al Dakkak, O. (2021). Building and benchmarking an Arabic speech commands dataset for small-footprint keyword spotting. Engineering Applications of Artificial Intelligence. Available: https://www.kaggle.com/datasets/abdulkaderghandoura/arabic-speech-commands-dataset

- **Turkish Speech Command Dataset**: 
  - Kurtkaya, M. (2021). Turkish speech command dataset. Available: https://www.kaggle.com/datasets/muratkurtkaya/turkish-speech-command-dataset

- **Kazakh, Tatar, and Russian Speech Commands**: 
  - Kuzdeuov, A., Nurgaliyev, S., Turmakhan, D., Laiyk, N., & Varol, H. A. (2023). Speech command recognition: Text-to-speech and speech corpus scraping are all you need. In Proc. of the International Conference on Robotics, Automation and Artificial Intelligence (RAAI).
  - Kuzdeuov, A., Gilmullin, R., Khakimov, B., & Varol, H. A. (2024). An open-source Tatar speech commands dataset for IoT and robotics applications. In Proc. of the Annual Conference of the IEEE Industrial Electronics Society (IECON).
  - Kuzdeuov, A., & Varol, H. A. (2025). Multilingual speech command recognition for voice controlled robots and smart systems. In 2025 11th International Conference on Control, Automation and Robotics (ICCAR).

These datasets provide a comprehensive foundation for developing and evaluating the multilingual SCR and LID models in this project.