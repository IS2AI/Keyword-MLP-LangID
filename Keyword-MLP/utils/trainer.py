import torch
from torch import nn, optim
from typing import Callable, Tuple
from torch.utils.data import DataLoader
from utils.misc import log, save_model
import os
import time
from tqdm import tqdm

##########################################
# Single-task code (unchanged)
##########################################

def train_single_batch(
    net: nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: Callable,
    device: torch.device
) -> Tuple[float, int]:
    """
    Performs a single training step for single-task classification.
    Returns (loss_value, number_of_correct).
    """
    data, targets = data.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(data)                  # shape [B, num_classes]
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = criterion(outputs, targets)   # single-task CE or label smoothing
    loss.backward()
    optimizer.step()

    correct = outputs.argmax(dim=1).eq(targets).sum().item()
    return loss.item(), correct

@torch.no_grad()
def evaluate_single(
    net: nn.Module,
    criterion: Callable,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Single-task evaluation (keyword classification only).
    Returns (accuracy, average_loss).
    """
    # Set model to eval mode
    net.eval()
    
    # Ensure deterministic behavior for CUDA operations
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    correct = 0
    running_loss = 0.0

    for batch in tqdm(dataloader):
        # Support both (data, targets) and (data, kw_label, lang_label) formats
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            data, targets = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            data, _, targets = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch) if isinstance(batch, (list, tuple)) else 'unknown'} elements")
        data, targets = data.to(device), targets.to(device)
        out = net(data)                          # [B, num_classes] for example
        loss = criterion(out, targets)
        running_loss += loss.item()

        preds = out.argmax(dim=1)
        correct += preds.eq(targets).sum().item()

    # Restore previous CUDA settings
    torch.backends.cudnn.deterministic = prev_deterministic
    torch.backends.cudnn.benchmark = prev_benchmark
    
    # Set model back to train mode
    net.train()
    
    accuracy = correct / len(dataloader.dataset)
    avg_loss = running_loss / len(dataloader)
    return accuracy, avg_loss

##########################################
# Multi-task code (new)
##########################################

def train_multitask_batch(
    net: nn.Module,
    data: torch.Tensor,
    kw_labels: torch.Tensor,
    lang_labels: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: Callable,
    device: torch.device
) -> Tuple[float, float, float, int, int]:
    """
    Performs a single training step for multi-task classification:
      - keywords (e.g. 35 classes)
      - language (e.g. 3 classes)
    Returns:
      (total_loss_value, kw_loss_value, lang_loss_value, num_correct_kw, num_correct_lang).
    """
    data = data.to(device)
    kw_labels = kw_labels.to(device)
    lang_labels = lang_labels.to(device)

    optimizer.zero_grad()
    logits_kw, logits_lang = net(data)  # model returns tuple
    total_loss = criterion(logits_kw, logits_lang, kw_labels, lang_labels)
    total_loss.backward()
    optimizer.step()

    # Get individual losses
    kw_loss, lang_loss = criterion.get_losses()

    # compute # correct for each head
    pred_kw = logits_kw.argmax(dim=1)
    correct_kw = pred_kw.eq(kw_labels).sum().item()

    pred_lang = logits_lang.argmax(dim=1)
    correct_lang = pred_lang.eq(lang_labels).sum().item()

    return total_loss.item(), kw_loss.item(), lang_loss.item(), correct_kw, correct_lang

@torch.no_grad()
def evaluate_multitask(
    net: nn.Module,
    criterion: Callable,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Multi-task evaluation for both keywords and language.
    Returns:
      (acc_keywords, acc_language, average_loss).
    """
    # Set model to eval mode
    net.eval()
    
    # Ensure deterministic behavior for CUDA operations
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    correct_kw = 0
    correct_lang = 0
    running_loss = 0.0
    total_samples = len(dataloader.dataset)

    for data, kw_labels, lang_labels in tqdm(dataloader):
        data = data.to(device)
        kw_labels = kw_labels.to(device)
        lang_labels = lang_labels.to(device)

        logits_kw, logits_lang = net(data)  # ( [B,35], [B,3] ) for example
        loss = criterion(logits_kw, logits_lang, kw_labels, lang_labels)
        running_loss += loss.item()

        # Count correct
        pred_kw = logits_kw.argmax(dim=1)
        correct_kw += pred_kw.eq(kw_labels).sum().item()

        pred_lang = logits_lang.argmax(dim=1)
        correct_lang += pred_lang.eq(lang_labels).sum().item()

    # Restore previous CUDA settings
    torch.backends.cudnn.deterministic = prev_deterministic
    torch.backends.cudnn.benchmark = prev_benchmark
    
    # Set model back to train mode
    net.train()
    
    avg_loss = running_loss / len(dataloader)
    acc_kw = correct_kw / total_samples
    acc_lang = correct_lang / total_samples
    return acc_kw, acc_lang, avg_loss

##########################################
# Combined training function
##########################################

def train(
    net: nn.Module,
    optimizer: optim.Optimizer,
    criterion: Callable,
    trainloader: DataLoader,
    valloader: DataLoader,
    schedulers: dict,
    config: dict
) -> None:
    """
    Main training loop that can handle single- OR multi-task training,
    depending on whether the input "criterion" and data loader yield
    single vs multi outputs.

    If "criterion" is MultiTaskLoss, we assume the loader yields
    (data, kw_label, lang_label). If single-task, we assume (data, targets).
    """
    step = 0
    best_acc = 0.0
    n_batches = len(trainloader)
    device = config["hparams"]["device"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

    # detect if multi-task by checking the type of criterion
    is_multitask = isinstance(criterion, nn.Module) and hasattr(criterion, "forward") and criterion.__class__.__name__ == "MultiTaskLoss"
    is_lang_only = config["hparams"]["model"].get("lang_only", False)

    for epoch in range(config["hparams"]["start_epoch"], config["hparams"]["n_epochs"]):
        t0 = time.time()
        running_loss = 0.0
        running_loss_kw = 0.0  # Track keyword loss
        running_loss_lang = 0.0  # Track language loss
        correct_main = 0
        correct_aux = 0  # used for language if multi-task

        net.train()

        for batch_index, batch_data in enumerate(trainloader):
            # warmup / scheduler steps
            if schedulers["warmup"] is not None and epoch < config["hparams"]["scheduler"]["n_warmup"]:
                schedulers["warmup"].step()
            elif schedulers["scheduler"] is not None:
                schedulers["scheduler"].step()

            # multi-task or single-task training step
            if is_multitask:
                # batch_data = (data, kw_label, lang_label)
                data, kw_lbl, lang_lbl = batch_data
                loss_val, kw_loss_val, lang_loss_val, corr_kw, corr_lang = train_multitask_batch(
                    net, data, kw_lbl, lang_lbl, optimizer, criterion, device
                )
                running_loss += loss_val
                running_loss_kw += kw_loss_val
                running_loss_lang += lang_loss_val
                correct_main += corr_kw
                correct_aux += corr_lang
            elif is_lang_only:
                # language-only single-head training step
                # batch_data = (data, kw_label, lang_label)
                data, _, lang_lbl = batch_data
                loss_val, corr_main = train_single_batch(
                    net, data, lang_lbl, optimizer, criterion, device
                )
                running_loss += loss_val
                correct_main += corr_main
            else:
                # batch_data = (data, targets)
                data, targets = batch_data
                loss_val, corr_main = train_single_batch(
                    net, data, targets, optimizer, criterion, device
                )
                running_loss += loss_val
                correct_main += corr_main

            if not step % config["exp"]["log_freq"]:
                if is_multitask:
                    log_dict = {
                        "epoch": epoch,
                        "loss": loss_val,
                        "loss_kw": kw_loss_val,
                        "loss_lang": lang_loss_val,
                        "lr": optimizer.param_groups[0]["lr"]
                    }
                else:
                    log_dict = {
                        "epoch": epoch,
                        "loss": loss_val,
                        "lr": optimizer.param_groups[0]["lr"]
                    }
                log(log_dict, step, config)

            step += 1

        # end of epoch
        time_per_epoch = time.time() - t0
        if is_multitask:
            # total samples
            total_samples = len(trainloader.dataset)
            # compute average training accuracy for keywords
            train_acc_kw = correct_main / total_samples
            # compute average training accuracy for language
            train_acc_lang = correct_aux / total_samples
            # compute average losses
            avg_loss = running_loss / len(trainloader)
            avg_loss_kw = running_loss_kw / len(trainloader)
            avg_loss_lang = running_loss_lang / len(trainloader)
            log_dict = {
                "epoch": epoch,
                "time_per_epoch": time_per_epoch,
                "train_acc_kw": train_acc_kw,
                "train_acc_lang": train_acc_lang,
                "avg_loss_per_ep": avg_loss,
                "avg_loss_kw_per_ep": avg_loss_kw,
                "avg_loss_lang_per_ep": avg_loss_lang
            }
        else:
            train_acc = correct_main / len(trainloader.dataset)
            log_dict = {
                "epoch": epoch,
                "time_per_epoch": time_per_epoch,
                "train_acc": train_acc,
                "avg_loss_per_ep": running_loss / len(trainloader)
            }
        log(log_dict, step, config)

        # validation
        if valloader is not None and not epoch % config["exp"]["val_freq"]:
            if is_multitask:
                from utils.trainer import evaluate_multitask
                val_acc_kw, val_acc_lang, avg_val_loss = evaluate_multitask(
                    net, criterion, valloader, device
                )
                # Get individual validation losses
                val_loss_kw, val_loss_lang = criterion.get_losses()
                # define a "main" val acc if you want to decide which is "best"  
                # e.g. let's treat keyword acc as the "main" metric
                val_acc = val_acc_kw
                log_dict = {
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                    "val_loss_kw": val_loss_kw.item(),
                    "val_loss_lang": val_loss_lang.item(),
                    "val_acc_kw": val_acc_kw,
                    "val_acc_lang": val_acc_lang
                }
            else:
                from utils.trainer import evaluate_single
                val_acc, avg_val_loss = evaluate_single(net, criterion, valloader, device)
                log_dict = {
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc
                }
            log(log_dict, step, config)

            # save best val ckpt
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(config["exp"]["save_dir"], "best.pth")
                save_model(epoch, val_acc, save_path, net, optimizer, schedulers["scheduler"], log_file)

    # final after training
    if valloader is not None:
        if is_multitask:
            val_acc_kw, val_acc_lang, avg_val_loss = evaluate_multitask(
                net, criterion, valloader, device
            )
            val_loss_kw, val_loss_lang = criterion.get_losses()
            val_acc = val_acc_kw
            log_dict = {
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "val_loss_kw": val_loss_kw.item(),
                "val_loss_lang": val_loss_lang.item(),
                "val_acc_kw": val_acc_kw,
                "val_acc_lang": val_acc_lang
            }
        else:
            val_acc, avg_val_loss = evaluate_single(net, criterion, valloader, device)
            log_dict = {
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "val_acc": val_acc
            }
        log(log_dict, step, config)
    
    # save final
    save_path = os.path.join(config["exp"]["save_dir"], "last.pth")
    save_model(epoch, valloader is not None and val_acc or 0, save_path, net, optimizer, schedulers["scheduler"], log_file)
