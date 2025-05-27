#!/usr/bin/env python
import os
import shutil
import json

# Base directories for each dataset 
EN_DATA = os.path.join('Keyword-MLP', 'data')       # English data
KK_DATA = os.path.join('Keyword-MLP', 'data_kk')    # Kazakh data
TT_DATA = os.path.join('Keyword-MLP', 'data_tt')    # Tatar data
RU_DATA = os.path.join('Keyword-MLP', 'data_ru')    # Russian data
AR_DATA = os.path.join('Keyword-MLP', 'data_ar')    # Arab data
TR_DATA = os.path.join('Keyword-MLP', 'data_tr')    # Turkish data

# Italian data path is: Keyword-MLP-ISSAI/lang_datasets/it/it/data_it
# IT_DATA = os.path.join('../lang_datasets', 'it', 'it', 'data_it')
# DE_DATA = os.path.join('../lang_datasets', 'de', 'de', 'data_de')
# ES_DATA = os.path.join('../lang_datasets', 'es', 'es', 'data_es')
FR_DATA = os.path.join('../lang_datasets', 'fr', 'fr', 'data_fr')
# NL_DATA = os.path.join('../lang_datasets', 'nl', 'nl', 'data_nl')
# PL_DATA = os.path.join('../lang_datasets', 'pl', 'pl', 'data_pl')
# FA_DATA = os.path.join('../lang_datasets', 'fa', 'fa', 'data_fa')
# RW_DATA = os.path.join('../lang_datasets', 'rw', 'rw', 'data_rw')
# CA_DATA = os.path.join('../lang_datasets', 'ca', 'ca', 'data_ca')

IT_DATA = os.path.join('../lang_datasets', 'data_it_aug')
DE_DATA = os.path.join('../lang_datasets', 'data_de_aug')
ES_DATA = os.path.join('../lang_datasets', 'data_es_aug')
# FR_DATA = os.path.join('../lang_datasets', 'data_fr_aug')
NL_DATA = os.path.join('../lang_datasets', 'data_nl_aug')
PL_DATA = os.path.join('../lang_datasets', 'data_pl_aug')
FA_DATA = os.path.join('../lang_datasets', 'data_fa_aug')
RW_DATA = os.path.join('../lang_datasets', 'data_rw_aug')
CA_DATA = os.path.join('../lang_datasets', 'data_ca_aug')

# Output directory for the unified dataset
OUT_DATA = os.path.join('Keyword-MLP', 'data_all')

def read_list_file(fname):
    """Read a text file and return a list of its lines."""
    with open(fname, "r") as f:
        lines = f.read().strip().splitlines()
    return lines

def process_line(line, expected_folder, lang_prefix):
    """
    Given a line like:
       "./data_kk/yes/484750822.wav"
    for expected_folder "data_kk" and lang_prefix "kk",
    remove the leading "./data_kk/" so that we have:
       "yes/484750822.wav"
    and then produce a new relative path with the language prefix added
       "yes/kk_484750822.wav"
    
    Returns:
        (original_rel, new_rel)
    where original_rel is used to copy the file from the source directory,
    and new_rel is the new relative path in the unified dataset.
    """
    line = line.strip()

    if line.startswith("./"):
        line = line[2:]
    elif line.startswith("."):
        line = line[1:]
    
    prefix_to_remove = expected_folder + "/"
    if line.startswith(prefix_to_remove):
        line = line[len(prefix_to_remove):]
    
    parts = line.split("/", 1)
    if len(parts) != 2:
        print("Malformed line:", line)
        return None, None
    cls, filename = parts
    original_rel = os.path.join(cls, filename)
    
    new_filename = f"{lang_prefix}_{filename}"
    new_rel = os.path.join(cls, new_filename)
    return original_rel, new_rel

def process_list(file_path, dataset_folder, lang_prefix):
    """
    Process a given text file (e.g. training_list.txt) for one dataset.
    Returns a list of tuples: (original_relative_path, new_relative_path)
    """
    lines = read_list_file(file_path)
    processed = []
    for line in lines:
        orig, new = process_line(line, dataset_folder, lang_prefix)
        if orig is not None:
            processed.append((orig, new))
    return processed

def copy_files(src_base, processed_list, dst_base):
    """
    For each tuple (orig_rel, new_rel) in processed_list, copy the file from:
         os.path.join(src_base, orig_rel)
    to:
         os.path.join(dst_base, new_rel)
    """
    for orig_rel, new_rel in processed_list:
        src_path = os.path.join(src_base, orig_rel)
        dst_path = os.path.join(dst_base, new_rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            shutil.copy(src_path, dst_path)
        except FileNotFoundError:
            print(f"File not found: {src_path}")

def write_combined_list(file_path, list_of_tuples):
    """
    Write a list file using the new relative paths from list_of_tuples.
    Each line will be prefixed with "./data_all/".
    """
    prefix = "./data_all/"
    lines = [prefix + new for (_, new) in list_of_tuples]
    with open(file_path, "w") as f:
        f.write("\n".join(lines))

def main():
    it_train = process_list(os.path.join(IT_DATA, "training_list.txt"), "data_it", "it")
    it_val   = process_list(os.path.join(IT_DATA, "validation_list.txt"), "data_it", "it")
    it_test  = process_list(os.path.join(IT_DATA, "testing_list.txt"), "data_it", "it")

    en_train = process_list(os.path.join(EN_DATA, "training_list.txt"), "data", "en")
    en_val   = process_list(os.path.join(EN_DATA, "validation_list.txt"), "data", "en")
    en_test  = process_list(os.path.join(EN_DATA, "testing_list.txt"), "data", "en")
    
    kk_train = process_list(os.path.join(KK_DATA, "training_list.txt"), "data_kk", "kk")
    kk_val   = process_list(os.path.join(KK_DATA, "validation_list.txt"), "data_kk", "kk")
    kk_test  = process_list(os.path.join(KK_DATA, "testing_list.txt"), "data_kk", "kk")
    
    tt_train = process_list(os.path.join(TT_DATA, "training_list.txt"), "data_tt", "tt")
    tt_val   = process_list(os.path.join(TT_DATA, "validation_list.txt"), "data_tt", "tt")
    tt_test  = process_list(os.path.join(TT_DATA, "testing_list.txt"), "data_tt", "tt")

    ru_train = process_list(os.path.join(RU_DATA, "training_list.txt"), "data_ru", "ru")
    ru_test  = process_list(os.path.join(RU_DATA, "testing_list.txt"), "data_ru", "ru")

    ar_train = process_list(os.path.join(AR_DATA, "training_list.txt"), "data_ar", "ar")
    ar_val   = process_list(os.path.join(AR_DATA, "validation_list.txt"), "data_ar", "ar")
    ar_test  = process_list(os.path.join(AR_DATA, "testing_list.txt"), "data_ar", "ar")

    tr_train = process_list(os.path.join(TR_DATA, "training_list.txt"), "data_tr", "tr")
    tr_val   = process_list(os.path.join(TR_DATA, "validation_list.txt"), "data_tr", "tr")
    tr_test  = process_list(os.path.join(TR_DATA, "testing_list.txt"), "data_tr", "tr")

    de_train = process_list(os.path.join(DE_DATA, "training_list.txt"), "data_de", "de")
    de_val   = process_list(os.path.join(DE_DATA, "validation_list.txt"), "data_de", "de")
    de_test  = process_list(os.path.join(DE_DATA, "testing_list.txt"), "data_de", "de")
    
    es_train = process_list(os.path.join(ES_DATA, "training_list.txt"), "data_es", "es")
    es_val   = process_list(os.path.join(ES_DATA, "validation_list.txt"), "data_es", "es")
    es_test  = process_list(os.path.join(ES_DATA, "testing_list.txt"), "data_es", "es")

    fr_train = process_list(os.path.join(FR_DATA, "training_list.txt"), "data_fr", "fr")
    fr_val   = process_list(os.path.join(FR_DATA, "validation_list.txt"), "data_fr", "fr")
    fr_test  = process_list(os.path.join(FR_DATA, "testing_list.txt"), "data_fr", "fr")

    pl_train = process_list(os.path.join(PL_DATA, "training_list.txt"), "data_pl", "pl")
    pl_val   = process_list(os.path.join(PL_DATA, "validation_list.txt"), "data_pl", "pl")
    pl_test  = process_list(os.path.join(PL_DATA, "testing_list.txt"), "data_pl", "pl")

    nl_train = process_list(os.path.join(NL_DATA, "training_list.txt"), "data_nl", "nl")
    nl_val   = process_list(os.path.join(NL_DATA, "validation_list.txt"), "data_nl", "nl")
    nl_test  = process_list(os.path.join(NL_DATA, "testing_list.txt"), "data_nl", "nl")

    fa_train = process_list(os.path.join(FA_DATA, "training_list.txt"), "data_fa", "fa")
    fa_val   = process_list(os.path.join(FA_DATA, "validation_list.txt"), "data_fa", "fa")
    fa_test  = process_list(os.path.join(FA_DATA, "testing_list.txt"), "data_fa", "fa")

    rw_train = process_list(os.path.join(RW_DATA, "training_list.txt"), "data_rw", "rw")
    rw_val   = process_list(os.path.join(RW_DATA, "validation_list.txt"), "data_rw", "rw")
    rw_test  = process_list(os.path.join(RW_DATA, "testing_list.txt"), "data_rw", "rw")

    ca_train = process_list(os.path.join(CA_DATA, "training_list.txt"), "data_ca", "ca")
    ca_val   = process_list(os.path.join(CA_DATA, "validation_list.txt"), "data_ca", "ca")
    ca_test  = process_list(os.path.join(CA_DATA, "testing_list.txt"), "data_ca", "ca")
    
    train_all = en_train + kk_train + tt_train + ru_train + ar_train + tr_train + it_train + de_train + es_train + fr_train + pl_train + nl_train + fa_train + rw_train + ca_train
    val_all   = en_val   + kk_val   + tt_val   + ar_val   + tr_val   + it_val + de_val + es_val + fr_val + pl_val + nl_val + fa_val + rw_val + ca_val
    test_all  = en_test  + kk_test  + tt_test  + ru_test  + ar_test  + tr_test + it_test + de_test + es_test + fr_test + pl_test + nl_test + fa_test + rw_test + ca_test
    
    copy_files(EN_DATA, en_train + en_val + en_test, OUT_DATA)
    copy_files(KK_DATA, kk_train + kk_val + kk_test, OUT_DATA)
    copy_files(TT_DATA, tt_train + tt_val + tt_test, OUT_DATA)
    copy_files(AR_DATA, ar_train + ar_val + ar_test, OUT_DATA)
    copy_files(TR_DATA, tr_train + tr_val + tr_test, OUT_DATA)
    copy_files(RU_DATA, ru_train + ru_test, OUT_DATA)
    copy_files(IT_DATA, it_train + it_val + it_test, OUT_DATA)
    copy_files(DE_DATA, de_train + de_val + de_test, OUT_DATA)
    copy_files(ES_DATA, es_train + es_val + es_test, OUT_DATA)
    copy_files(FR_DATA, fr_train + fr_val + fr_test, OUT_DATA)
    copy_files(PL_DATA, pl_train + pl_val + pl_test, OUT_DATA)
    copy_files(NL_DATA, nl_train + nl_val + nl_test, OUT_DATA)
    copy_files(FA_DATA, fa_train + fa_val + fa_test, OUT_DATA)
    copy_files(RW_DATA, rw_train + rw_val + rw_test, OUT_DATA)
    copy_files(CA_DATA, ca_train + ca_val + ca_test, OUT_DATA)
    
    os.makedirs(OUT_DATA, exist_ok=True)
    write_combined_list(os.path.join(OUT_DATA, "training_list.txt"), train_all)
    write_combined_list(os.path.join(OUT_DATA, "validation_list.txt"), val_all)
    write_combined_list(os.path.join(OUT_DATA, "testing_list.txt"), test_all)
    
    # Print the number of lines in each combined file
    for split in ["training_list.txt", "validation_list.txt", "testing_list.txt"]:
        file_path = os.path.join(OUT_DATA, split)
        with open(file_path, "r") as f:
            num_lines = sum(1 for _ in f)
        print(f"{split}: {num_lines} lines")
    
    shutil.copy(os.path.join(EN_DATA, "label_map.json"),
                os.path.join(OUT_DATA, "label_map.json"))
    
    print("All data combined successfully into:", OUT_DATA)
    print("You can now train with: python train.py --conf configs/config_mul_lc.yaml")

if __name__ == "__main__":
    main()