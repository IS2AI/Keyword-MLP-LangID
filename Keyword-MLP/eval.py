import torch
import yaml
from argparse import ArgumentParser
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json

from utils.dataset import get_loader
from utils.misc import seed_everything, get_model
from utils.trainer import evaluate_single, evaluate_multitask

def load_config(conf_path):
    with open(conf_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_deterministic():
    # Set PyTorch deterministic flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    g = torch.Generator()
    g.manual_seed(0)
    
    return seed_worker, g

def detect_dataset_type(test_list, lang_map):
    """
    Determine if this is a single language dataset or a combined dataset.
    Returns:
        tuple: (is_single_language, language_code or None)
    """
    if not test_list:
        return False, None
    sample_path = test_list[0]
    # Check if it's a single language dataset
    for lang_code in lang_map.keys():
        if 'data_{}/'.format(lang_code) in sample_path:
            return True, lang_code
    # Check if it's a combined dataset with language prefixes
    filename = os.path.basename(sample_path)
    if '_' in filename and len(filename) > 2:
        prefix = filename.split('_')[0]
        if prefix in lang_map:
            return False, None  # Combined dataset, no default language needed
    return False, None  # Unknown or standard combined dataset

def extract_language_from_path(file_path, lang_map):
    """Extract language code from file path or name using lang_map."""
    for lang_code in lang_map.keys():
        if 'data_{}/'.format(lang_code) in file_path:
            return lang_code
    filename = os.path.basename(file_path)
    if '_' in filename and len(filename) > 2:
        prefix = filename.split('_')[0]
        if prefix in lang_map:
            return prefix
    return None  # Unable to determine language

def main(args):
    # 1) Load config
    config = load_config(args.conf)
    # Load lang_map from config if present, else from default path
    if "lang_map" in config:
        with open(config["lang_map"], "r") as f:
            lang_map = json.load(f)
    else:
        with open("lang_map.json", "r") as f:
            lang_map = json.load(f)
    # Set fixed seed and ensure deterministic behavior
    seed_everything(config["hparams"]["seed"])
    seed_worker, g = set_deterministic()
    # Update dataloader config for deterministic behavior
    if "dataloader" not in config["hparams"]:
        config["hparams"]["dataloader"] = {}
    config["hparams"]["dataloader"]["worker_init_fn"] = seed_worker
    config["hparams"]["dataloader"]["generator"] = g
    config["hparams"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Determine evaluation mode flags
    multitask = config["hparams"]["model"].get("multitask", False)
    lang_only = config["hparams"]["model"].get("lang_only", False)
    # 2) Build model
    model = get_model(config["hparams"]["model"])
    model = model.to(config["hparams"]["device"])
    # 3) Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=config["hparams"]["device"])
    # Strip 'module.' prefix from checkpoint keys if loaded from DataParallel
    raw_sd = ckpt["model_state_dict"]
    new_sd = {}
    for k, v in raw_sd.items():
        new_sd[k.replace("module.", "") if k.startswith("module.") else k] = v
    model.load_state_dict(new_sd)
    model.eval()
    # 4) Prepare test list 
    with open(args.test_list, "r") as f:
        test_list = f.read().strip().split("\n")
    # Detect if this is a single language dataset or a combined dataset
    is_single_lang, default_lang = detect_dataset_type(test_list, lang_map)
    if is_single_lang:
        print("Detected single language dataset ({})".format(default_lang))
    else:
        print("Detected combined multi-language dataset")
    # Create data loader with appropriate language handling
    testloader = get_loader(test_list, config, train=False, default_lang=default_lang)

    # 5) Determine evaluation mode and set loss function
    if multitask:
        from utils.loss import MultiTaskLoss
        criterion = MultiTaskLoss(w_kw=1.0, w_lang=1.0)
    elif lang_only:
        # loss for language-only classification
        if config["hparams"].get("l_smooth", 0):
            from utils.loss import LabelSmoothingLoss
            criterion = LabelSmoothingLoss(
                num_classes=config["hparams"]["model"]["num_langs"],
                smoothing=config["hparams"]["l_smooth"]
            )
        else:
            import torch.nn as nn
            criterion = nn.CrossEntropyLoss()
    else:
        # loss for keyword-only classification
        if config["hparams"].get("l_smooth", 0):
            from utils.loss import LabelSmoothingLoss
            criterion = LabelSmoothingLoss(
                num_classes=config["hparams"]["model"]["num_classes"],
                smoothing=config["hparams"]["l_smooth"]
            )
        else:
            import torch.nn as nn
            criterion = nn.CrossEntropyLoss()
    
    # 6) Evaluate overall performance (multi-task, language-only, or keyword-only)
    all_preds = []
    all_labels = []
    lang_info = []  # Will store language for each sample
    
    if multitask:
        test_acc_kw, test_acc_lang, test_loss = evaluate_multitask(
            model, criterion, testloader, config["hparams"]["device"]
        )
        print("\nOverall Test results on {}:".format(args.test_list))
        print("  Test Loss:         {:.5f}".format(test_loss))
        print("  Test Keyword Acc:  {:.5f}".format(test_acc_kw))
        print("  Test Language Acc: {:.5f}".format(test_acc_lang))
    elif lang_only:
        # Language-only evaluation
        test_acc, test_loss = evaluate_single(
            model, criterion, testloader, config["hparams"]["device"]
        )
        print("\nOverall Test results on {}:".format(args.test_list))
        print("  Test Loss:          {:.5f}".format(test_loss))
        print("  Test Language Acc:  {:.5f}".format(test_acc))
        # Per-language stats for language-only model
        file_to_lang = {}
        for file_path in test_list:
            lang_str = extract_language_from_path(file_path, lang_map)
            if lang_str is None:
                lang_str = "unknown"
            file_to_lang[os.path.normpath(file_path)] = lang_str

        all_preds = []
        all_labels = []
        lang_info = []

        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(testloader):
                # unpack (data, kw_label, lang_label)
                data, _, lang_labels = batch
                # get file paths for current batch
                batch_start = i * testloader.batch_size
                batch_end = min(batch_start + testloader.batch_size, len(test_list))
                batch_files = test_list[batch_start:batch_end]
                # record language for each file
                batch_langs = [file_to_lang[os.path.normpath(f)] for f in batch_files]
                lang_info.extend(batch_langs)

                # forward pass
                data = data.to(config["hparams"]["device"])
                lang_labels = lang_labels.to(config["hparams"]["device"])
                outputs = model(data)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lang_labels.cpu().numpy())

        # Compute per-language accuracy
        per_language = {}
        for idx, lang in enumerate(lang_info):
            if lang not in per_language:
                per_language[lang] = {"total": 0, "correct": 0}
            per_language[lang]["total"] += 1
            if all_preds[idx] == all_labels[idx]:
                per_language[lang]["correct"] += 1

        print("\nPer-language language accuracy:")
        for lang, stats in sorted(per_language.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            print(f"  Language '{lang}': Total Samples: {stats['total']}")
            print(f"    Correct: {stats['correct']}")
            print(f"    Language Accuracy: {acc:.5f}")
            print("-" * 40)
    else:
        # Get language info for each test file
        file_to_lang = {}
        for file_path in test_list:
            lang_str = extract_language_from_path(file_path, lang_map)
            if lang_str is None:
                lang_str = "unknown"
            file_to_lang[os.path.normpath(file_path)] = lang_str
            
        # Use custom evaluation to collect per-language metrics
        total_correct = 0
        total_samples = 0
        running_loss = 0.0
        
        with torch.no_grad():
            model.eval()
            for i, (data, labels) in enumerate(testloader):
                # Get file paths for current batch
                batch_start = i * testloader.batch_size
                batch_end = min(batch_start + testloader.batch_size, len(test_list))
                batch_files = test_list[batch_start:batch_end]
                
                # Get language for each file in batch
                batch_langs = [file_to_lang.get(os.path.normpath(f), "unknown") for f in batch_files]
                lang_info.extend(batch_langs)
                
                # Forward pass
                data = data.to(config["hparams"]["device"])
                labels = labels.to(config["hparams"]["device"])
                
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                # Calculate metrics
                running_loss += loss.item() * data.size(0)
                predictions = outputs.argmax(dim=1)
                batch_correct = (predictions == labels).sum().item()
                total_correct += batch_correct
                total_samples += labels.size(0)
                
                # Store predictions and labels
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate overall metrics
        test_loss = running_loss / total_samples
        test_acc = total_correct / total_samples
        
        print("\nOverall Test results on {}:".format(args.test_list))
        print("  Test Loss: {:.5f}".format(test_loss))
        print("  Test Acc:  {:.5f}".format(test_acc))
        
        # Verify we have the same number of samples
        assert len(all_preds) == len(all_labels) == len(lang_info), \
            "Mismatch in collected data: preds={}, labels={}, langs={}".format(len(all_preds), len(all_labels), len(lang_info))
        
        # Process per-language metrics
        is_multilang_test = 'data_all' in args.test_list or not is_single_lang
        if is_multilang_test:
            print("\nComputing per-language keyword accuracy...")
        
            # Calculate per-language metrics
            per_language = {}
            for i, lang in enumerate(lang_info):
                if lang not in per_language:
                    per_language[lang] = {"total": 0, "correct": 0}
                
                per_language[lang]["total"] += 1
                if all_preds[i] == all_labels[i]:
                    per_language[lang]["correct"] += 1
            
            # Print per-language results
            print("\nPer-language keyword accuracy:")
            weighted_correct = 0
            for lang, stats in sorted(per_language.items()):
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                print("  Language '{}': Total Samples: {}".format(lang, stats['total']))
                print("    Correct: {}".format(stats['correct']))
                print("    Keyword Accuracy: {:.5f}".format(acc))
                print("-" * 40)
                weighted_correct += stats["correct"]
            
            # Calculate weighted average to verify
            overall_samples = sum(stats["total"] for stats in per_language.values())
            weighted_avg = weighted_correct / overall_samples if overall_samples > 0 else 0.0
            print("\nWeighted average keyword accuracy: {:.5f}".format(weighted_avg))
            assert abs(weighted_avg - test_acc) < 1e-5, \
                "Calculation error: weighted avg ({:.5f}) != reported acc ({:.5f})".format(weighted_avg, test_acc)
    
    # 7) Compute per-language metrics for multitask model
    if config["hparams"]["model"].get("multitask", False):
        # For multitask models - existing code
        per_language = {}  # To hold metrics for each language
        # Build a reverse language map from dataset (e.g., {0: 'en', 1: 'kk', ...})
        reverse_lang_map = {v: k for k, v in lang_map.items()}
        
        # Lists for confusion matrix
        all_true_langs = []
        all_pred_langs = []
        
        # Iterate over the testloader to accumulate per-language stats
        with torch.no_grad():
            model.eval()  # Ensure model is in eval mode
            
            for data, kw_labels, lang_labels in testloader:
                data = data.to(config["hparams"]["device"])
                kw_labels = kw_labels.to(config["hparams"]["device"])
                lang_labels = lang_labels.to(config["hparams"]["device"])
                
                # Get model predictions
                logits_kw, logits_lang = model(data)
                pred_kw = logits_kw.argmax(dim=1)
                pred_lang = logits_lang.argmax(dim=1)
                
                # Collect language predictions for confusion matrix
                all_true_langs.extend(lang_labels.cpu().numpy())
                all_pred_langs.extend(pred_lang.cpu().numpy())
                
                # Process each sample in the batch
                for i in range(len(lang_labels)):
                    # Ground truth language (numeric)
                    lang_idx = lang_labels[i].item()
                    # Get language string using reverse mapping
                    lang_str = reverse_lang_map.get(lang_idx, "unknown({})".format(lang_idx))
                    
                    if lang_str not in per_language:
                        per_language[lang_str] = {"total": 0, "correct_kw": 0, "correct_lang": 0}
                    
                    per_language[lang_str]["total"] += 1
                    if pred_kw[i].item() == kw_labels[i].item():
                        per_language[lang_str]["correct_kw"] += 1
                    if pred_lang[i].item() == lang_labels[i].item():
                        per_language[lang_str]["correct_lang"] += 1

        print("\nPer-language results:")
        for lang, stats in sorted(per_language.items()):
            total = stats["total"]
            acc_kw = stats["correct_kw"] / total if total > 0 else 0.0
            acc_lang = stats["correct_lang"] / total if total > 0 else 0.0
            print("  Language '{}': Total Samples: {}".format(lang, total))
            print("    Keyword Accuracy:  {:.5f}".format(acc_kw))
            print("    Language Accuracy: {:.5f}".format(acc_lang))
            print("-" * 40)
        
        # Calculate and print confusion matrix
        all_true_langs = np.array(all_true_langs)
        all_pred_langs = np.array(all_pred_langs)
        
        # Ordered labels
        sorted_indices = sorted(reverse_lang_map.keys())
        labels = [reverse_lang_map[i] for i in sorted_indices]

        # Raw counts
        cm = confusion_matrix(all_true_langs, all_pred_langs, labels=sorted_indices)

        # Row‐normalize to percentages
        cm_row = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        print("\nLanguage ID Confusion Matrix (row %):")
        header = " " * 12 + "".join(f"{l:8s}" for l in labels)
        print(header)
        for i, row in enumerate(cm_row):
            print(f"{labels[i]:12s}" + "".join(f"{v:8.1f}" for v in row))

        # Overall accuracy
        acc = np.trace(cm) / cm.sum() * 100
        print(f"\nOverall accuracy: {acc:.2f}%")

        if args.out_fig:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm_row, interpolation='nearest', cmap=plt.cm.Blues)
            
            ax.set(
                xticks=np.arange(len(labels)),
                yticks=np.arange(len(labels)),
                xticklabels=labels,
                yticklabels=labels,
                xlabel="Predicted",
                ylabel="Actual",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
            plt.setp(ax.get_yticklabels(), fontsize=16)
            ax.set_xlabel("Predicted", fontsize=18)
            ax.set_ylabel("Actual", fontsize=18)

            thresh = cm_row.max() / 2.0
            for i in range(cm_row.shape[0]):
                for j in range(cm_row.shape[1]):
                    ax.text(
                        j, i,
                        f"{cm_row[i, j]:.1f}",
                        ha="center", va="center",
                        fontsize=16,
                        color="white" if cm_row[i, j] > thresh else "black"
                    )

            fig.tight_layout()

            plt.savefig(args.out_fig, dpi=300, bbox_inches="tight")
            print(f"Saved row‐normalized confusion matrix to {args.out_fig}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conf", type=str, required=True, help="Path to config_all.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pth or last.pth")
    parser.add_argument("--test_list", type=str, required=True, help="Path to test_list.txt to evaluate on")
    parser.add_argument("--out_fig", type=str, default="", help="Output path for the confusion matrix figure")
    args = parser.parse_args()
    main(args)