import numpy as np
import pdb
import json
from scipy.signal import butter, filtfilt
import os
from tqdm import tqdm



def get_info_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    width = data['video_width']
    height = data['video_height']
    cam = data['cam']
    return width, height, cam

def denormalize_pose_img(pose):
    res_w, res_h, _ = get_info_from_json(json_path)
    pose_denorm = pose.copy()
    for idx in range(len(pose)):
        pose_denorm[idx, :, :2] = (pose[idx, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
        pose_denorm[idx, :, 2:] = pose[idx, :,2:] * res_w / 2
    return pose_denorm

def normalize_pose_img(pose):
    res_w, res_h, _ = get_info_from_json(json_path)
    pose_norm = pose.copy()
    for idx in range(len(pose)):
        pose_norm[idx, :, :2] = pose[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
        pose_norm[idx, :, 2:] = pose_norm[idx, :, 2:] / res_w * 2
    return pose_norm

def camera_to_image_frame(pose3d, box, camera, rootIdx):
    rectangle_3d_size = 2000.0
    ratio = (abs(box[2] - box[0]) + 1) / rectangle_3d_size
    pose3d_image_frame = np.zeros_like(pose3d)
    pose3d_image_frame[:, :2] = _weak_project(
        pose3d.copy(), camera['fx'], camera['fy'], camera['cx'], camera['cy'])
    pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
    pose3d_image_frame[:, 2] = pose3d_depth
    return pose3d_image_frame

def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d

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

def _retrieve_camera(cameras_path, json_path):
    _ , _, cam_name = get_info_from_json(json_path)
    with open(cameras_path, 'r') as f:
        cameras = json.load(f)
    vid_cam = cameras[cam_name]
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
    camera['name'] = cam_name
    return camera

def world_to_camera_frame(pose3d, R, xyz):
    R = np.array(R)
    R[1:, :] *= -1
    pose3d = pose3d - np.array(xyz)
    pose3d = (R @ pose3d.T).T
    return pose3d

def _infer_box(pose3d, camera, rootIdx):
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
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])

def calculate_speed(pose3d, fps):
    speed = np.zeros((pose3d.shape[0], pose3d.shape[1]))
    dt = 1 / fps  # Time step

    for i in range(1, pose3d.shape[0]):
        speed[i] = np.linalg.norm(pose3d[i] - pose3d[i-1], axis=1) / dt  # Divide by dt

    #drop the first frame
    speed = speed[1:]
    #swap the axes
    speed = np.swapaxes(speed, 0, 1)
    return speed

def coco_to_h36m(pose):
    """
    Converts COCO pose format (17 keypoints) to H36M format (17 keypoints),
    based on the provided mapping.

    Args:
        pose (numpy.ndarray): Pose data of shape (seq, 17, 3), where `seq` is the sequence length,
                               17 is the number of keypoints, and 3 represents (x, y, confidence).
    
    Returns:
        numpy.ndarray: Pose data in H36M format with shape (seq, 17, 3).
    """
    # Define the mapping from COCO keypoints to H36M keypoints
    coco_to_h36m_mapping = {
        0: [12, 11],   
        1: 12,        
        2: 14,        
        3: 16,       
        4: 11,        
        5: 13,        
        6: 15,       
        7: [6, 5, 12, 11], 
        8: [6,5],       
        9: [6,5,0,0], 
        10: 0,      
        11: 5,       
        12: 7,       
        13: 9,       
        14: 6,       
        15: 8,       
        16: 10        
    }

    # Initialize the output array for H36M pose
    h36m_pose = np.zeros_like(pose)

    # Apply the mapping to each frame in the sequence
    for i in range(pose.shape[0]):  # Iterate through each frame in the sequence
        for coco_idx, h36m_idx in coco_to_h36m_mapping.items():
            # If the mapping is to a list, we take the average of the keypoints
            if isinstance(h36m_idx, list):
                # h36m_pose = np.mean(pose[i, h36m_idx], axis=0)
                h36m_pose[i, coco_idx] = np.mean(pose[i, h36m_idx], axis=0)
            else:
                h36m_pose[i, coco_idx] = pose[i, h36m_idx]
    return h36m_pose

def get_fps(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    fps = data['fps']
    return fps

def calculate_joint_angle(pose):
    """
    Compute joint angles for a sequence of 3D poses.

    Parameters:
    - pose: NumPy array of shape (T, 17, 3), where T is the number of frames.

    Returns:
    - angles: NumPy array of shape (T, 4), containing angles in radians for each frame.
    """
    # Define keypoints for each angle: right leg, left leg, left arm, right arm
    keypoints = [(1, 2, 3), (4, 5, 6), (11, 12, 13), (14, 15, 16)]
    
    # Initialize array for angles
    angles = np.zeros((pose.shape[0], len(keypoints)))
    
    for j, (p1, p2, p3) in enumerate(keypoints):
        vec1 = pose[:, p1] - pose[:, p2]  # Vector 1
        vec2 = pose[:, p3] - pose[:, p2]  # Vector 2

        norm1 = np.linalg.norm(vec1, axis=1)
        norm2 = np.linalg.norm(vec2, axis=1)

        # Compute cosine similarity safely
        cos_theta = np.sum(vec1 * vec2, axis=1) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Prevent numerical errors

        # Compute angles
        angles[:, j] = np.arccos(cos_theta)
    #swap the axes to get the shape (4, T)
    angles = np.swapaxes(angles, 0, 1)
    return angles 

if __name__ == '__main__':

    npz_root = './tcpformer/test_set_results' 
    pose_2d_root = './moganet/test_set_results'
    gt_root = './AthletePose3D/data/test_set'

    save_path = './test_set_results/kinematics/'
    cameras_path = './AthletePose3D/cam_param.json'
    rootIdx = 0

    joint_pose_cam_vel_list = []
    joint_pose_img_vel_list = []
    joint_pose_2d_vel_list = []
    joint_gt_2d_vel_list = []
    joint_gt_3d_img_vel_list = []
    joint_gt_3d_cam_vel_list = []
    joint_gt_3d_world_vel_list = []

    angle_cam_list = []
    angle_img_list = []
    angle_2d_list = []
    angle_gt_2d_list = []
    angle_gt_3d_img_list = []
    angle_gt_3d_cam_list = []
    angle_gt_3d_world_list = []

    npz_list = os.listdir(npz_root)
    for npz_name in tqdm(npz_list, desc='Processing'):
        npz_path = os.path.join(npz_root, npz_name)
        pose_2d_path = os.path.join(pose_2d_root, npz_name)
        subject = npz_name.split('_')[0]
        action_name_list = npz_name.split('_')[1:]
        action_name = '_'.join(action_name_list).replace('.npz', '')
        gt_path = os.path.join(gt_root, subject, f'{action_name}_h36m.npy')
        json_path = os.path.join(gt_root, subject, f'{action_name}.json')

        fps = int(get_fps(json_path))
        camera = _retrieve_camera(cameras_path, json_path)

        pose_2d = np.load(pose_2d_path, allow_pickle=True)['reconstruction']
        pose_2d = coco_to_h36m(pose_2d)
        pose_2d = pose_2d[:, :, :2]
        pose = np.load(npz_path)
        pose = pose['reconstruction']
        pose_img = pose.copy()
        pose_img = denormalize_pose_img(pose)

        #get gt data
        gt = np.load(gt_path)
        gt_3d_world = gt.copy()
        gt_3d_cam_list = []
        gt_3d_img_list = []
        pose_cam_list = []
        for idx in range(len(gt_3d_world)):
            gt_3d_world_i = gt_3d_world[idx]
            gt_3d_cam = world_to_camera_frame(gt_3d_world_i, camera['R'], camera['T'])
            box = _infer_box(gt_3d_cam, camera, rootIdx)
            gt_3d_img = camera_to_image_frame(gt_3d_cam, box, camera, rootIdx)
            pose_cam = image_to_camera_frame(pose_img[idx].copy(), box, camera, rootIdx, root_depth=0)
            gt_3d_cam_list.append(gt_3d_cam)
            gt_3d_img_list.append(gt_3d_img)
            pose_cam_list.append(pose_cam)
        gt_3d_cam = np.array(gt_3d_cam_list)
        gt_3d_img = np.array(gt_3d_img_list)
        gt_2d = gt_3d_img[:, :, :2]
        pose_cam = np.array(pose_cam_list)

        #get the kpt velocity of pose

        fs = fps 
        cutoff = 8
        order = 4
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        pose_cam_vel = calculate_speed(pose_cam, fps)
        pose_img_vel = calculate_speed(pose_img, fps)
        pose_2d_vel = calculate_speed(pose_2d, fps)
        gt_2d_vel = calculate_speed(gt_2d, fps)
        gt_3d_img_vel = calculate_speed(gt_3d_img, fps)
        gt_3d_cam_vel = calculate_speed(gt_3d_cam, fps)
        gt_3d_world_vel = calculate_speed(gt_3d_world, fps)    

        angle_cam = calculate_joint_angle(pose_cam)
        angle_img = calculate_joint_angle(pose_img)
        angle_2d = calculate_joint_angle(pose_2d)
        angle_gt_2d = calculate_joint_angle(gt_2d)
        angle_gt_3d_img = calculate_joint_angle(gt_3d_img)
        angle_gt_3d_cam = calculate_joint_angle(gt_3d_cam)
        angle_gt_3d_world = calculate_joint_angle(gt_3d_world)
        
        #apply filter
        pose_cam_vel_filtered = filtfilt(b, a, pose_cam_vel, axis=1)
        pose_img_vel_filtered = filtfilt(b, a, pose_img_vel, axis=1)
        pose_2d_vel_filtered = filtfilt(b, a, pose_2d_vel, axis=1)
        gt_2d_vel_filtered = filtfilt(b, a, gt_2d_vel, axis=1)
        gt_3d_img_vel_filtered = filtfilt(b, a, gt_3d_img_vel, axis=1)
        gt_3d_cam_vel_filtered = filtfilt(b, a, gt_3d_cam_vel, axis=1)
        gt_3d_world_vel_filtered = filtfilt(b, a, gt_3d_world_vel, axis=1)

        angle_cam_filtered = filtfilt(b, a, angle_cam, axis=1)
        angle_img_filtered = filtfilt(b, a, angle_img, axis=1)
        angle_2d_filtered = filtfilt(b, a, angle_2d, axis=1)
        angle_gt_2d_filtered = filtfilt(b, a, angle_gt_2d, axis=1)
        angle_gt_3d_img_filtered = filtfilt(b, a, angle_gt_3d_img, axis=1)
        angle_gt_3d_cam_filtered = filtfilt(b, a, angle_gt_3d_cam, axis=1)
        angle_gt_3d_world_filtered = filtfilt(b, a, angle_gt_3d_world, axis=1)

        joint_pose_cam_vel_list.append(pose_cam_vel_filtered)
        joint_pose_img_vel_list.append(pose_img_vel_filtered)
        joint_pose_2d_vel_list.append(pose_2d_vel_filtered)
        joint_gt_2d_vel_list.append(gt_2d_vel_filtered)
        joint_gt_3d_img_vel_list.append(gt_3d_img_vel_filtered)
        joint_gt_3d_cam_vel_list.append(gt_3d_cam_vel_filtered)
        joint_gt_3d_world_vel_list.append(gt_3d_world_vel_filtered)

        angle_cam_list.append(angle_cam_filtered)
        angle_img_list.append(angle_img_filtered)
        angle_2d_list.append(angle_2d_filtered)
        angle_gt_2d_list.append(angle_gt_2d_filtered)
        angle_gt_3d_img_list.append(angle_gt_3d_img_filtered)
        angle_gt_3d_cam_list.append(angle_gt_3d_cam_filtered)
        angle_gt_3d_world_list.append(angle_gt_3d_world_filtered)

    out_dict = {'joint_pose_cam_vel_list': joint_pose_cam_vel_list,
    'joint_pose_img_vel_list': joint_pose_img_vel_list,
    'joint_pose_2d_vel_list': joint_pose_2d_vel_list,
    'joint_gt_2d_vel_list': joint_gt_2d_vel_list,
    'joint_gt_3d_img_vel_list': joint_gt_3d_img_vel_list,
    'joint_gt_3d_cam_vel_list': joint_gt_3d_cam_vel_list,
    'joint_gt_3d_world_vel_list': joint_gt_3d_world_vel_list,
    'angle_cam_list': angle_cam_list,
    'angle_img_list': angle_img_list,
    'angle_2d_list': angle_2d_list,
    'angle_gt_2d_list': angle_gt_2d_list,
    'angle_gt_3d_img_list': angle_gt_3d_img_list,
    'angle_gt_3d_cam_list': angle_gt_3d_cam_list,
    'angle_gt_3d_world_list': angle_gt_3d_world_list}

    #save as pickle
    import pickle
    with open(save_path + 'pose_velocity.pkl', 'wb') as f:
        pickle.dump(out_dict, f)
    print('done')
