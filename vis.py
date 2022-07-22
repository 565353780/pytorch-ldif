#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess

class LargeVis(object):
    def __init__(self, large_vis_path, threads):
        self.large_vis_path = large_vis_path
        self.threads = threads
        return

    def runCMD(self, cmd):
        result = subprocess.check_output(cmd, shell=True)
        if result is None:
            print("[ERROR][LargeVis::runCMD]")
            print("\t check_output failed!")
            return False
        return True

    def transDataTo2D(self, file_path):
        if not os.path.exists(file_path):
            print("[ERROR][LargeVis::transDataTo2D]")
            print("\t file not exist!")
            return False

        output_file_path = file_path.replace(".txt", "_2d.txt")
        if os.path.exists(output_file_path):
            return True

        cmd = self.large_vis_path + \
            " -input " + file_path + \
            " -output " + output_file_path + \
            " -threads " + str(self.threads)
        if not self.runCMD(cmd):
            print("[ERROR][LargeVis::transDataTo2D]")
            print("\t runCMD failed!")
            return False

        return True

    def visualFile(self, file_path):
        print("[INFO][LargeVis::visualFile]")
        print("\t start transDataTo2D for " + file_path + "...")

        if not self.transDataTo2D(file_path):
            print("[ERROR][LargeVis::visualFile]")
            print("\t transDataTo2D failed!")
            return False

        output_file_path = file_path.replace(".txt", "_plot.png")
        if os.path.exists(output_file_path):
            return True

        cmd = "python ./Method/plot.py" + \
            " -input " + file_path.replace(".txt", "_2d.txt") + \
            " -label " + file_path.replace(".txt", "_label.txt") + \
            " -output " + file_path.replace(".txt", "_plot")
        if not self.runCMD(cmd):
            print("[ERROR][LargeVis::visualFile]")
            print("\t runCMD failed!")
            return False
        return True

def demo():
    large_vis = LargeVis("/home/chli/github/LargeVis/Linux/LargeVis", 24)

    large_vis.visualFile("./out/LargeVis/train_image_ldif.txt")
    large_vis.visualFile("./out/LargeVis/train_sdf_ldif.txt")
    large_vis.visualFile("./out/LargeVis/train_image_sdf_ldif.txt")

    large_vis.visualFile("./out/LargeVis/val_image_ldif.txt")
    large_vis.visualFile("./out/LargeVis/val_sdf_ldif.txt")
    large_vis.visualFile("./out/LargeVis/val_image_sdf_ldif.txt")

    large_vis.visualFile("./out/LargeVis/train_val_image_ldif.txt")
    large_vis.visualFile("./out/LargeVis/train_val_sdf_ldif.txt")
    large_vis.visualFile("./out/LargeVis/train_val_image_sdf_ldif.txt")
    return True

if __name__ == "__main__":
    demo()

