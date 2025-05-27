#!/usr/bin/env python
import os
import glob
import json
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_parser import get_config
from utils.misc import get_model
from utils.dataset import GoogleSpeechDataset


@torch.no_grad()
def get_preds_multitask(net, dataloader, device) -> list:
    """
    Inference for multi-task models: net(x) -> (logits_kw, logits_lang).
    Returns a list of (kw_pred, lang_pred) for each sample in the dataset.
    """
    net.eval()
    preds_list = []

    for batch in tqdm(dataloader, desc="Inferring"):
        # If the dataset returns only x, we get shape [B, 1, 40, 98].
        # If it returns (x, ...), do: `data = batch[0]`.
        if isinstance(batch, (list, tuple)):
            data = batch[0]  # in case dataset returns (x,) or (x, kw_lbl, lang_lbl) – not typical for inference
        else:
            data = batch
        data = data.to(device)

        logits_kw, logits_lang = net(data)
        pred_kw = logits_kw.argmax(dim=1).cpu().numpy().tolist()
        pred_lang = logits_lang.argmax(dim=1).cpu().numpy().tolist()

        # Collect predictions
        for kw, lang in zip(pred_kw, pred_lang):
            preds_list.append((kw, lang))
    return preds_list


@torch.no_grad()
def get_preds_single(net, dataloader, device) -> list:
    """
    Inference for single-task models: net(x) -> logits.
    Returns a list of numeric predictions, one per sample.
    """
    net.eval()
    preds_list = []

    for batch in tqdm(dataloader, desc="Inferring"):
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch
        data = data.to(device)

        out = net(data)
        preds = out.argmax(dim=1).cpu().numpy().tolist()
        preds_list.extend(preds)
    return preds_list


def main(args):
    # Load config
    config = get_config(args.conf)

    # Build model
    model = get_model(config["hparams"]["model"])

    # Load weights
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    # Collect input file paths
    if os.path.isdir(args.inp):
        data_list = glob.glob(os.path.join(args.inp, "*.wav"))
    elif os.path.isfile(args.inp):
        data_list = [args.inp]
    else:
        raise ValueError(f"Invalid input path: {args.inp}")

    # Decide device
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    model = model.to(device)

    # Are we in multi-task mode?
    multitask = config["hparams"]["model"].get("multitask", False)

    # --------------------------------------------
    # Optionally load label maps for final labeling
    # --------------------------------------------
    # If these are provided, we can map numeric predictions to strings.
    inv_kw_map = None
    inv_lang_map = {0: "EN", 1: "KK", 2: "TT"}  # default if no --langmap

    if args.lmap:
        with open(args.lmap, "r") as f:
            # e.g. { "0": "backward", "1": "bed", ... }
            kw_label_map = json.load(f)
        # invert -> { "backward":0, "bed":1 } but we need numeric->string:
        # so we do { 0:"backward", 1:"bed", ... }
        inv_kw_map = {int(k): v for k, v in kw_label_map.items()}

    if args.langmap:
        with open(args.langmap, "r") as f2:
            # e.g. { "en": 0, "kk":1, "tt":2 } but we want numeric->string
            # let's say the file is { "0":"en", "1":"kk", "2":"tt" } or the other way around
            # whichever you prefer. If your file is { "en":0, "kk":1, "tt":2 }, invert it.
            loaded_langmap = json.load(f2)

        # If loaded_langmap is e.g. { "en":0, "kk":1, "tt":2 },
        # invert it to {0:"en", 1:"kk", 2:"tt"}:
        inv_lang_map = {}
        for k, v in loaded_langmap.items():
            # k might be "en", v might be 0
            # so we do inv_lang_map[v] = k
            inv_lang_map[v] = k

    # --------------------------------------------
    # Build a dataset that returns only the audio x
    # (no label_map / lang_map needed for inference)
    # --------------------------------------------
    dataset = GoogleSpeechDataset(
        data_list=data_list,
        label_map=None,   # <--- no label_map for inference
        lang_map=None,    # <--- no lang_map for inference
        audio_settings=config["hparams"]["audio"],
        aug_settings=None,
        cache=0
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Inference
    if multitask:
        preds = get_preds_multitask(model, dataloader, device)
    else:
        preds = get_preds_single(model, dataloader, device)

    # Convert numeric preds to strings if we have label maps
    pred_dict = {}
    if multitask:
        # preds is list of (kw_pred, lang_pred)
        for file_path, (kw_idx, lang_idx) in zip(data_list, preds):
            if inv_kw_map:
                kw_str = inv_kw_map.get(kw_idx, str(kw_idx))
            else:
                kw_str = str(kw_idx)
            lang_str = inv_lang_map.get(lang_idx, str(lang_idx))
            pred_dict[file_path] = {"keyword": kw_str, "language": lang_str}
    else:
        # single task
        for file_path, p in zip(data_list, preds):
            if inv_kw_map:
                kw_str = inv_kw_map.get(p, str(p))
            else:
                kw_str = str(p)
            pred_dict[file_path] = kw_str

    # Save to JSON
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "preds.json")
    with open(out_path, "w") as f:
        json.dump(pred_dict, f, indent=4)
    print(f"Saved preds to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conf", type=str, required=True, help="Path to config (audio params, model architecture, etc).")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument("--inp", type=str, required=True, help="Input .wav file or folder of .wav files.")
    parser.add_argument("--out", type=str, default="./", help="Folder to write preds.json.")
    parser.add_argument("--lmap", type=str, default=None, help="Path to label_map.json for keywords.")
    parser.add_argument("--langmap", type=str, default=None, help="Path to JSON for language mapping.")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda.")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    assert os.path.exists(args.inp), f"Could not find input {args.inp}"
    main(args)
