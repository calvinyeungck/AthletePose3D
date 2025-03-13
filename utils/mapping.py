import json
from typing import List, Tuple
import numpy as np
import pandas as pd
import pdb

def load_rig_mapping(rig_file: str, rig_name: str, marker_labels: List[str]) -> Tuple[List[str], List[List[int]]]:
    #ref https://github.com/ryota-skating/FS-Jump3D
    with open(rig_file, "r") as f:
        rig_data = json.load(f)
    
    joint_names = list(rig_data[rig_name].keys())
    marker_idxs = [[marker_labels.index(label) for label in rig_data[rig_name][joint]] for joint in joint_names]

    return joint_names, marker_idxs

def apply_rig_format(pose3d, joint_names, marker_idxs) -> np.ndarray:
    #ref https://github.com/ryota-skating/FS-Jump3D
    formatted_pose3d = np.zeros((len(joint_names), 3))
    for i, idxs in enumerate(marker_idxs):
        formatted_pose3d[i, :] = np.mean(pose3d[idxs, :], axis=0)
    return formatted_pose3d

def get_marker_labels_c3d_json(c3d_json):
    #check if c3d_json is a path or a dict
    if isinstance(c3d_json, str):
        with open(c3d_json, "r") as f:
            c3d_data = json.load(f)
    else:
        c3d_data = c3d_json
  
    marker_names = []
    for marker_i in range(len(c3d_data['Markers'])):
        marker_names.append(c3d_data['Markers'][marker_i]['Name'])
    
    return marker_names

def get_marker_labels_tsv(tsv_file):
    df = pd.read_csv(tsv_file, sep="\t", skiprows=9, nrows=1, header=None)
    marker_names = df.iloc[0].to_list()[1:]
    return marker_names

def check_all_pt_in_frame(pose3d, frame):
    #check if all the 3D points are in the frame
    for pt in pose3d:
        if pt[0] < 0 or pt[0] >= frame.shape[1] or pt[1] < 0 or pt[1] >= frame.shape[0]:
            return False
    return True

