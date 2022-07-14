#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

def getModelPath(config):
    log_dict = config['log']
    if 'resume_path' not in log_dict.keys():
        return None

    model_resume_path = log_dict['resume_path']
    if model_resume_path[-1] != "/":
        model_resume_path += "/"

    last_model_path = model_resume_path + "model_last.pth"
    if not os.path.exists(last_model_path):
        return None
    return last_model_path

