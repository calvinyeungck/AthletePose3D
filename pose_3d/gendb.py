# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import argparse
import os
import os.path as osp
import scipy.io as sio
import numpy as np
# import parse
# import cameras
import pickle
import json
import pdb
from tqdm import tqdm


def find_train_val_dirs(dataset_root_dir):
    train_dirs = os.listdir(osp.join(dataset_root_dir, 'train_set'))
    val_dirs = os.listdir(osp.join(dataset_root_dir, 'valid_set'))
    return train_dirs, val_dirs


def infer_meta_from_name(datadir):
    #read the json file 
    with open(datadir, 'r') as f:
        info = json.load(f)
    if 'rotation_ccw_90' in info.keys():
        rotation = bool(info['rotation_ccw_90'])
    else:
        rotation = False
    meta = {
        'width': info['video_width'],
        'height': info['video_height'],
        'action': info['type'],
        'subaction': info['source_file'].replace('.mp4', ''),
        'camera': info['cam'],
        'rotation': rotation,
        'fps': info['fps'],
        'subject': datadir.split('/')[-2]
    }
    return meta


def load_db(dataset_root_dir, dset, vid, cams, rootIdx=0):
    annofile = dataset_root_dir+"/"+dset.replace('.mp4', '_h36m.npy')
    anno = np.load(annofile, allow_pickle=True)
    numimgs = len(anno)
    joints_3d_world = anno
    meta = infer_meta_from_name(dataset_root_dir+"/"+dset.replace('.mp4', '.json'))
    cam = _retrieve_camera(cams, meta['camera'])
    split = "train_img" if "train" in dataset_root_dir else "valid_img"
    dataset = []
    for i in range(numimgs):
        joint_3d_world = joints_3d_world[i]
        joint_3d_cam = world_to_camera_frame(joint_3d_world, cam['R'], cam['T'])

        box = _infer_box(joint_3d_cam, cam, rootIdx, meta)
        joint_3d_image = camera_to_image_frame(joint_3d_cam, box, cam, rootIdx, meta)
        if meta['rotation']:
            video_width = meta['height']
            video_height = meta['width']
        else:
            video_width = meta['width']
            video_height = meta['height']
        center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
        scale = ((abs(box[2] - box[0])) / 200.0, (abs(box[3] - box[1])) / 200.0)
        ratio = (abs(box[2] - box[0]) + 1) / 2000.0 # 2000 is the rectangle size
        dataitem = {
            'videoid': vid,
            'cameraid': meta['camera'],
            'camera_param': cam,
            'joint_3d_image': joint_3d_image,
            'joint_3d_camera': joint_3d_cam,
            'center': center,
            'scale': scale,
            'ratio': ratio,
            'box': box,
            'subject': meta['subject'],
            'action': meta['action'],
            'subaction': meta['subaction'],
            'root_depth': joint_3d_cam[rootIdx, 2],
            'video_width': video_width,
            'video_height': video_height,
            'rotation': meta['rotation'],
            'fps': meta['fps'],
        }

        dataset.append(dataitem)
    return dataset

def world_to_camera_frame(pose3d, R, xyz):
    R = np.array(R)
    R[1:, :] *= -1
    pose3d = pose3d - np.array(xyz)
    pose3d = (R @ pose3d.T).T
    return pose3d

def camera_to_image_frame(pose3d, box, camera, rootIdx, meta):
    rectangle_3d_size = 2000.0
    ratio = (abs(box[2] - box[0]) + 1) / rectangle_3d_size
    pose3d_image_frame = np.zeros_like(pose3d)
    pose3d_image_frame[:, :2] = _weak_project(
        pose3d.copy(), camera['fx'], camera['fy'], camera['cx'], camera['cy'])
    pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
    pose3d_image_frame[:, 2] = pose3d_depth
    # if meta['rotation'] == True then the image is rotated 90 degrees
    if meta['rotation']:
        # Flip Y and store the flipped value
        flipped_y = meta['height'] - pose3d_image_frame[:, 1]

        # Swap coordinates
        pose3d_image_frame[:, 1] = pose3d_image_frame[:, 0]
        pose3d_image_frame[:, 0] = flipped_y
        pose3d_image_frame[:, 1] = meta['width']/2 - (pose3d_image_frame[:, 1] - meta['width']/2)
        pose3d_image_frame[:, 0] = meta['height']/2 - (pose3d_image_frame[:, 0] - meta['height']/2)
        
    return pose3d_image_frame


def image_to_camera_frame(pose3d_image_frame, box, camera, rootIdx, root_depth):
    rectangle_3d_size = 2000.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_image_frame[:, 2] = pose3d_image_frame[:, 2] / ratio + root_depth

    cx, cy, fx, fy = camera['cx'], camera['cy'], camera['fx'], camera['fy']
    pose3d_image_frame[:, 0] = (pose3d_image_frame[:, 0] - cx) / fx
    pose3d_image_frame[:, 1] = (pose3d_image_frame[:, 1] - cy) / fy
    pose3d_image_frame[:, 0] *= pose3d_image_frame[:, 2]
    pose3d_image_frame[:, 1] *= pose3d_image_frame[:, 2]
    return pose3d_image_frame


def _infer_box(pose3d, camera, rootIdx, meta):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[:2] -= 1000.0
    br_joint = root_joint.copy()
    br_joint[:2] += 1000.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))
    

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    root_joint = np.reshape(root_joint, (1, 3))
    if meta['rotation']:
        tl2d = [meta['height'] - tl2d[1], tl2d[0]]
        br2d = [meta['height'] - br2d[1], br2d[0]]
        #rotate the coordinates
        tl2d[0] = meta['height']/2 - (tl2d[0] - meta['height']/2)
        tl2d[1] = meta['width']/2 - (tl2d[1] - meta['width']/2)
        br2d[0] = meta['height']/2 - (br2d[0] - meta['height']/2)
        br2d[1] = meta['width']/2 - (br2d[1] - meta['width']/2)
        #swap tl2d and br2d
        tl2d, br2d = br2d, tl2d
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d


def _retrieve_camera(cameras, cameraidx):
    vid_cam = cameras[cameraidx]
    # Intrinsics
    fx = vid_cam['affine_intrinsics_matrix'][0][0]  # Focal length in x
    fy = vid_cam['affine_intrinsics_matrix'][1][1]  # Focal length in y
    cx = vid_cam['affine_intrinsics_matrix'][0][2]  # Principal point x
    cy = vid_cam['affine_intrinsics_matrix'][1][2]  # Principal point y

    # Distortion coefficients
    k = vid_cam['distortion'][:3]  # Radial distortion
    p = vid_cam['distortion'][3:]  # Tangential distortion
    camera = {}
    camera['R'] = vid_cam['extrinsic_matrix']
    camera['T'] = vid_cam['xyz']
    camera['fx'] = fx
    camera['fy'] = fy
    camera['cx'] = cx
    camera['cy'] = cy
    camera['k'] = k
    camera['p'] = p
    camera['name'] = cameraidx
    return camera


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_root_dir = './AthletePose3D/data'
    cams_path = "./AthletePose3D/cam_param.json"
    with open(cams_path, 'r') as f:
        cams = json.load(f)

    train_dirs, val_dirs = find_train_val_dirs(args.dataset_root_dir)

    video_count = 0
    db_train = []
    for folder in train_dirs:
        video_list = os.listdir(args.dataset_root_dir+"/train_set/"+folder)
        #filter out non-mp4 files
        video_list = [x for x in video_list if x.endswith('.mp4')]
        for video in tqdm(video_list, desc= f"Processing {folder}"):
            if np.mod(video_count, 1) == 0:
                print('Process {}: {}'.format(video_count,  folder+"/"+video))

            data = load_db(args.dataset_root_dir+"/train_set/"+folder, video, video_count, cams)
            db_train.extend(data)
            video_count += 1

    video_count = 0
    db_valid = []
    for folder in val_dirs:
        video_list = os.listdir(args.dataset_root_dir+"/valid_set/"+folder)
        #filter out non-mp4 files
        video_list = [x for x in video_list if x.endswith('.mp4')]
        for video in tqdm(video_list, desc= f"Processing {folder}"):
            if np.mod(video_count, 1) == 0:
                print('Process {}: {}'.format(video_count, folder+"/"+video))

            data = load_db(args.dataset_root_dir+"/valid_set/"+folder, video, video_count, cams)
            db_valid.extend(data)
            video_count += 1


    datasets = {'train': db_train, 'validation': db_valid}
    with open('./AthletePose3D/pose_3d/train.pkl', 'wb') as f:
        pickle.dump(datasets['train'], f)

    with open('./AthletePose3D/pose_3d/valid.pkl', 'wb') as f:
        pickle.dump(datasets['validation'], f)
