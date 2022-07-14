#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy as np

def read_obj(model_path, flags = ('v')):
    fid = open(model_path, 'r')

    data = {}
    for head in flags:
        data[head] = []

    for line in fid:
        line = re.split('\s+', line.strip())
        if line[0] in flags:
            data[line[0]].append(line[1:])
    fid.close()

    if 'v' in data.keys():
        data['v'] = np.array(data['v']).astype(float)

    if 'vt' in data.keys():
        data['vt'] = np.array(data['vt']).astype(float)

    if 'vn' in data.keys():
        data['vn'] = np.array(data['vn']).astype(float)
    return data

def write_obj(objfile, data):
    with open(objfile, 'w+') as file:
        for item in data['v']:
            file.write('v' + ' %f' * len(item) % tuple(item) + '\n')

        for item in data['f']:
            file.write('f' + ' %s' * len(item) % tuple(item) + '\n')
    return True

def det(a):
    return a[0][0]*a[1][1]*a[2][2] + \
        a[0][1]*a[1][2]*a[2][0] + \
        a[0][2]*a[1][0]*a[2][1] - \
        a[0][2]*a[1][1]*a[2][0] - \
        a[0][1]*a[1][0]*a[2][2] - \
        a[0][0]*a[1][2]*a[2][1]

def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    if magnitude == 0.:
        return (0., 0., 0.)
    return (x/magnitude, y/magnitude, z/magnitude)

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)

def get_area(poly):
    if len(poly) < 3:
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]

    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

def calculate_face_area(data):
    face_areas = []

    for face in data['f']:
        vid_in_face = [int(item.split('/')[0]) for item in face]
        face_area = get_area(data['v'][np.array(vid_in_face) - 1,:3].tolist())
        face_areas.append(face_area)
    return face_areas

def sample_pnts_from_obj(data, n_pnts = 5000, mode = 'uniform'):
    flags = data.keys()

    all_pnts = data['v'][:,:3]

    area_list = np.array(calculate_face_area(data))
    distribution = area_list/np.sum(area_list)

    new_pnts = []
    if mode == 'random':
        random_face_ids = np.random.choice(len(data['f']), n_pnts, replace=True, p=distribution)
        random_face_ids, sample_counts = np.unique(random_face_ids, return_counts=True)

        for face_id, sample_count in zip(random_face_ids, sample_counts):
            face = data['f'][face_id]

            vid_in_face = [int(item.split('/')[0]) for item in face]

            weights = np.diff(np.sort(np.vstack(
                [np.zeros((1, sample_count)),
                 np.random.uniform(0, 1, size=(len(vid_in_face) - 1, sample_count)),
                 np.ones((1, sample_count))]), axis=0), axis=0)

            new_pnt = all_pnts[np.array(vid_in_face) - 1].T.dot(weights)

            if 'vn' in flags:
                nid_in_face = [int(item.split('/')[2]) for item in face]
                new_normal = data['vn'][np.array(nid_in_face)-1].T.dot(weights)
                new_pnt = np.hstack([new_pnt, new_normal])


            new_pnts.append(new_pnt.T)
        return np.vstack(new_pnts)

    for face_idx, face in enumerate(data['f']):
        vid_in_face = [int(item.split('/')[0]) for item in face]

        n_pnts_on_face = distribution[face_idx] * n_pnts

        if n_pnts_on_face < 1:
            continue

        dim = len(vid_in_face)
        npnts_dim = (np.math.factorial(dim - 1)*n_pnts_on_face)**(1/(dim-1))
        npnts_dim = int(npnts_dim)

        weights = np.stack(np.meshgrid(*[np.linspace(0, 1, npnts_dim) for _ in range(dim - 1)]), 0)
        weights = weights.reshape(dim - 1, -1)
        last_column = 1 - weights.sum(0)
        weights = np.vstack([weights, last_column])
        weights = weights[:, last_column >= 0]

        new_pnt = (all_pnts[np.array(vid_in_face) - 1].T.dot(weights)).T

        if 'vn' in flags:
            nid_in_face = [int(item.split('/')[2]) for item in face]
            new_normal = data['vn'][np.array(nid_in_face) - 1].T.dot(weights)
            new_pnt = np.hstack([new_pnt, new_normal])

        new_pnts.append(new_pnt)
    return np.vstack(new_pnts)

def normalize_to_unit_square(points):
    centre = (points.max(0) + points.min(0))/2.
    point_shapenet = points - centre

    scale = point_shapenet.max()
    point_shapenet = point_shapenet / scale
    return point_shapenet, centre, scale

def normalize(input_path, output_folder):
    output_path = os.path.join(output_folder, 'mesh_normalized.obj')

    obj_data = read_obj(input_path, ['v', 'f'])
    obj_data['v'] = normalize_to_unit_square(obj_data['v'])[0]
    write_obj(output_path, obj_data)
    return output_path

def remove_if_exists(f):
    if os.path.exists(f):
        os.remove(f)
    return True

