#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

def getValidFolderPath(folder_path):
    valid_folder_path = folder_path
    if valid_folder_path[-1] != "/":
        valid_folder_path += "/"

    if not os.path.exists(valid_folder_path):
        print("[ERROR][paths::getValidFolderPath]")
        print("\t folder not exist!")
        print("\t " + folder_path)
        return None
    return valid_folder_path

def getModelPath(config, mode):
    model_file_name_dict = {"train": "model_last.pth",
                            "test": "model_best.pth"}

    log_dict = config['log']
    if 'resume_path' not in log_dict.keys():
        return None

    model_resume_path = log_dict['resume_path']
    if model_resume_path[-1] != "/":
        model_resume_path += "/"

    last_model_path = model_resume_path + model_file_name_dict[mode]
    if not os.path.exists(last_model_path):
        return None
    return last_model_path

