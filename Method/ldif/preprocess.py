#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import tqdm
import subprocess
import numpy as np
from PIL import Image
from multiprocessing import Pool

from Config.configs import PIX3DConfig, ShapeNetConfig

from Method.preprocess import normalize, remove_if_exists

class PreProcesser(object):
    def __init__(self, config):
        self.del_intermediate_result = True
        self.skip_done = True
        self.processes = 12
        self.scale_norm = 0.25
        self.bbox_half = 0.7
        self.neighbors = 30

        self.gaps_folder_path = "./external/ldif/gaps/bin/x86_64/"
        self.mesh_fusion_folder_path = "./external/mesh_fusion/"

        self.config = config

        self.bbox = ' '.join([str(-self.bbox_half), ] * 3 + [str(self.bbox_half), ] * 3)
        self.spacing = self.bbox_half * 2 / 32
        self.output_root = self.config.root_path + "ldif/"

        self.not_valid_model_list = []
        return

    def runCMD(self, cmd):
        return subprocess.check_output(cmd, shell=True, env={"LIBGL_ALWAYS_INDIRECT": "0"})

    def make_output_folder(self, mesh_path):
        mesh_unit_path = mesh_path.split(self.config.mesh_folder)[1]
        mesh_class_name = mesh_unit_path.split("/")[0]
        output_mesh_folder_name = mesh_unit_path.split(mesh_class_name + "/")[1].replace(".obj", "").replace("/", ".")
        output_folder = self.output_root + mesh_class_name + "/" + output_mesh_folder_name + "/"

        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def make_watertight(self, input_path, output_folder):
        output_path = output_folder + "mesh_orig.obj"

        off_path = os.path.splitext(output_path)[0] + '.off'
        cmd = 'xvfb-run -a -s "-screen 0 800x600x24" ' + \
            'meshlabserver ' + '-i ' + input_path + ' -o ' + off_path
        self.runCMD(cmd)

        cmd = sys.executable + ' ' + self.mesh_fusion_folder_path + 'scale.py' + \
            ' --in_file ' + off_path + \
            ' --out_dir ' + output_folder + \
            ' --t_dir ' + output_folder + \
            ' --overwrite'
        self.runCMD(cmd)

        # create depth maps
        cmd = 'xvfb-run -a -s "-screen 0 800x600x24" ' + \
            sys.executable + ' ' + self.mesh_fusion_folder_path + 'fusion.py' + \
            ' --mode=render' + \
            ' --in_file ' + off_path + \
            ' --out_dir ' + output_folder + \
            ' --overwrite'
        self.runCMD(cmd)

        # produce watertight mesh
        depth_path = off_path + '.h5'
        transform_path = os.path.splitext(output_path)[0] + '.npz'
        cmd = sys.executable + ' ' + self.mesh_fusion_folder_path + 'fusion.py' + \
            ' --mode=fuse' + \
            ' --in_file ' + depth_path + \
            ' --out_dir ' + output_folder + \
            ' --t_dir ' + output_folder + \
            ' --overwrite'
        self.runCMD(cmd)

        os.remove(off_path)
        os.remove(transform_path)
        os.remove(depth_path)
        return output_path

    def processImage(self, sample):
        output_folder = self.make_output_folder(self.config.metadata_path + sample['model'])

        img_name = os.path.splitext(os.path.split(sample['img'])[1])[0]
        output_path = output_folder + img_name + '.npy'
        if not self.skip_done or not os.path.exists(output_path):
            img = np.array(Image.open(self.config.metadata_path + sample['img']).convert('RGB'))
            img = img[sample['bbox'][1]:sample['bbox'][3], sample['bbox'][0]:sample['bbox'][2]]
            np.save(output_path, img)

        img_name = os.path.splitext(os.path.split(sample['mask'])[1])[0]
        output_path = output_folder + img_name + '_mask.npy'
        if not self.skip_done or not os.path.exists(output_path):
            img = np.array(Image.open(self.config.metadata_path + sample['mask']).convert('L'))
            img = img[sample['bbox'][1]:sample['bbox'][3], sample['bbox'][0]:sample['bbox'][2]]
            np.save(output_path, img)
        return True

    def processMesh(self, mesh_path):
        output_folder = self.make_output_folder(mesh_path)
        if self.skip_done and os.path.exists(output_folder + 'uniform_points.sdf'):
            return True

        normalized_obj = normalize(mesh_path, output_folder)
        watertight_obj = self.make_watertight(normalized_obj, output_folder)

        normalized_ply = os.path.splitext(normalized_obj)[0] + '.ply'
        cmd = 'xvfb-run -a -s "-screen 0 800x600x24" ' + \
            'meshlabserver -i ' + normalized_obj + ' -o ' + normalized_ply
        self.runCMD(cmd)

        watertight_ply = os.path.splitext(watertight_obj)[0] + '.ply'
        try:
            cmd = 'xvfb-run -a -s "-screen 0 800x600x24" ' + \
                'meshlabserver -i ' + watertight_obj + ' -o ' + watertight_ply
            self.runCMD(cmd)
        except:
            self.not_valid_model_list.append(watertight_obj)
            os.remove(normalized_obj)
            os.remove(normalized_ply)
            os.remove(watertight_obj)
            print("[ERROR][PreProcesser::processMesh]")
            print("\t watertight_obj [" + watertight_obj + "] file not valid!")
            return False

        scaled_ply = output_folder + "scaled_watertight.ply"
        cmd = self.gaps_folder_path + 'msh2msh ' + \
            watertight_ply + ' ' + scaled_ply + \
            ' -scale_by_pca' + \
            ' -translate_by_centroid' + \
            ' -scale ' + str(self.scale_norm) + \
            ' -debug_matrix ' + output_folder + '/orig_to_gaps.txt'
        os.system(cmd)

        cmd = self.gaps_folder_path + 'msh2df ' + \
            scaled_ply + ' ' + output_folder + '/coarse_grid.grd' + \
            ' -bbox ' + str(self.bbox) + \
            ' -border 0' + \
            ' -spacing ' + str(self.spacing) + \
            ' -estimate_sign'
        os.system(cmd)

        cmd = self.gaps_folder_path + 'msh2pts ' + \
            scaled_ply + ' ' + output_folder + '/nss_points.sdf' + \
            ' -near_surface' + \
            ' -max_distance ' + str(self.spacing) + \
            ' -num_points 100000' + \
            ' -binary_sdf'
        os.system(cmd)

        cmd = self.gaps_folder_path + 'msh2pts ' + \
            scaled_ply + ' ' + output_folder + '/uniform_points.sdf' + \
            ' -uniform_in_bbox' + \
            ' -bbox ' + str(self.bbox) + \
            ' -npoints 100000' + \
            ' -binary_sdf'
        os.system(cmd)

        if self.del_intermediate_result:
            remove_if_exists(normalized_obj)
            remove_if_exists(watertight_obj)
            remove_if_exists(scaled_ply)
        return True

    def processAllImage(self):
        print('Processing imgs...')
        with open(self.config.metadata_file, 'r') as file:
            metadata = json.load(file)

        with open(self.config.train_split, 'r') as file:
            splits = json.load(file)

        with open(self.config.test_split, 'r') as file:
            splits += json.load(file)

        ids = [int(os.path.basename(file).split('.')[0]) for file in splits if 'flipped' not in file]
        samples = [metadata[id] for id in ids]

        with Pool(self.processes) as p:
            r = list(tqdm.tqdm(p.imap(self.processImage, samples), total=len(samples)))
        return True

    def processAllMesh(self):
        mesh_path_list = []
        for root, dirs, files in os.walk(self.config.mesh_folder):
            for file in files:
                if file[-4:] != ".obj":
                    continue
                mesh_path_list.append(root + "/" + file)

        with Pool(self.processes) as p:
            r = list(tqdm.tqdm(p.imap(self.processMesh, mesh_path_list), total=len(mesh_path_list)))
        return True

    def process(self):
        if not self.processAllImage():
            print("[ERROR][PreProcesser::process]")
            print("\t processAllImage failed!")
            return False
        if not self.processAllMesh():
            print("[ERROR][PreProcesser::process]")
            print("\t processAllMesh failed!")
            return False

        if len(self.not_valid_model_list) > 0:
            print("[WARN][PreProcesser::process]")
            print("\t not valid model list:")
            for path in self.not_valid_model_list:
                print("\t\t " + path)
        return True

def demo():
    config = ShapeNetConfig()

    preprocesser = PreProcesser(config)
    preprocesser.processAllMesh()
    return True

if __name__ == '__main__':
    demo()

