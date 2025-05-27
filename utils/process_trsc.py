#!/usr/bin/env python3
"""
utils/process_trsc.py

Process the Turkish Speech Command dataset:

1) Remove `dataBase.xlsx` from raw directory.
2) Rename the raw `database/` folder to `data_tr/`.
3) Keep only the nine command folders; rename them to English names:
   "asagi"→"down", "dur"→"stop", "evet"→"yes",
   "geri"→"backward", "hayir"→"no", "ileri"→"forward",
   "sag"→"right", "sol"→"left", "yukari"→"up".
   Delete all other folders.
4) Split `.wav` files by speaker ID into train/dev/test according to provided lists.
5) Write three lists (`training_list.txt`, `validation_list.txt`, `testing_list.txt`) under `data_tr/`,
   with entries like:
     ./data_tr/yes/evet_AKZI_BVSWLFP.wav
   Sorted by speaker ID, ending with an empty line.

Usage:
  python utils/process_trsc.py \
    --data_root data/turkish-speech-command-dataset \
    --raw_subdir database
"""
import argparse
import os
import shutil
from pathlib import Path

# Mapping Turkish folder → English
T2E = {
    "asagi": "down",
    "dur":   "stop",
    "evet":  "yes",
    "geri":  "backward",
    "hayir": "no",
    "ileri": "forward",
    "sag":   "right",
    "sol":   "left",
    "yukari":"up",
}
# Speakers split lists
TEST_SPK = {
    'AKZI','APTE','BEUS','DJUT','DXEF','DZWQ','FDOY','HOFC','INMC','JDKS','JQLB',
    'MGRC','MYAL','NUID','OZRD','POLE','PUSH','QMTA','TRDK','UFAH','UKMV','VTCN',
    'WEQJ','WTDE','XKBI','YMHE','ZWDC'
}
VAL_SPK = {
    'BJOY','BKYJ','CXUL','DPMF','ENLP','FRBA','GJRH','GKYU','JXCP','LGNH','MAIW',
    'MDUS','MJON','OFNT','QDHZ','RNMT','SWBQ','URLW','VGIZ','VOBW','VTXW','WIPK',
    'WXEK','ZDHR','ZXKY'
}
# All other speakers are training


def main():
    parser = argparse.ArgumentParser(description="Process Turkish dataset splits")
    parser.add_argument(
        '--data_root', type=Path,
        default=Path('data/turkish-speech-command-dataset'),
        help='root containing the raw subdir'
    )
    parser.add_argument(
        '--raw_subdir', type=str,
        default='database',
        help='folder with Turkish folders and files'
    )
    args = parser.parse_args()

    raw_dir = args.data_root / args.raw_subdir
    if not raw_dir.is_dir():
        raise RuntimeError(f"Raw subdir not found: {raw_dir}")

    # 1) remove dataBase.xlsx
    xlsx = raw_dir / 'dataBase.xlsx'
    if xlsx.exists():
        xlsx.unlink()
        print(f"Removed: {xlsx}")

    # 2) rename raw_dir → data_tr
    data_tr = raw_dir.parent / 'data_tr'
    if data_tr.exists():
        raise RuntimeError(f"Target exists already: {data_tr}")
    raw_dir.rename(data_tr)
    print(f"Renamed {raw_dir} → {data_tr}")

    # 3) rename and prune command folders
    for child in list(data_tr.iterdir()):
        if child.is_dir():
            name = child.name
            if name in T2E:
                new_dir = data_tr / T2E[name]
                child.rename(new_dir)
                print(f"Renamed folder: {name} → {new_dir.name}")
            else:
                shutil.rmtree(child)
                print(f"Removed folder: {name}")

    # Prepare lists
    train_lines = []
    val_lines   = []
    test_lines  = []

    # 4) iterate files
    for cmd in sorted(T2E.values()):
        folder = data_tr / cmd
        if not folder.is_dir():
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith('.wav'):
                continue
            # speaker ID is second underscore-separated field
            parts = fn.split('_')
            if len(parts) < 2:
                continue
            spk = parts[1]
            path = f"./data_tr/{cmd}/{fn}"
            if spk in TEST_SPK:
                test_lines.append(path)
            elif spk in VAL_SPK:
                val_lines.append(path)
            else:
                # assume training if not in test/val
                train_lines.append(path)

    # sort by speaker id only (stable sort preserves per-speaker order)
    def by_spk(p): return p.split('/')[-1].split('_')[1]
    train_lines.sort(key=by_spk)
    val_lines.sort(key=by_spk)
    test_lines.sort(key=by_spk)

    # 5) write outputs with trailing blank line
    for fname, lines in [
        ('training_list.txt',    train_lines),
        ('validation_list.txt',  val_lines),
        ('testing_list.txt',     test_lines),
    ]:
        out_p = data_tr / fname
        out_p.write_text("\n".join(lines) + "\n", encoding='utf-8')
        print(f"Wrote {fname}: {len(lines)} lines")

if __name__ == '__main__':
    main()
