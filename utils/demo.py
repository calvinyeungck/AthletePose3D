import numpy as np
import json
import cv2
import pdb
import mapping as mp
import vis
import hashlib


def get_cam_param(cam_param_path, json_path):
    with open(cam_param_path, "r") as f:
        cam_param = json.load(f)
    with open(json_path, "r") as f:
        data = json.load(f)
    cam_param = cam_param[data['cam']]

    return cam_param

def get_marker_labels(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data['keypoint_name']

def project_2d(keypoint_data, cam_param, frame, undistort=False, adj_cam=False,rotate=False):
    """
    Projects 3D keypoints onto a 2D image plane based on camera parameters.

    Args:
        keypoint_data (list): A list of 3D keypoints (N x 3 array) to be projected.
        cam_param (dict): Camera parameters, including intrinsic, extrinsic matrices, and distortion coefficients.
        frame (ndarray): The image frame on which to draw the projected keypoints.
        undistort (bool): Flag indicating whether to undistort the points. Default is False.
        adj_cam (bool): Flag to adjust the camera position (scale it to meters). Default is False.

    Returns:
        ndarray: The frame with the projected keypoints drawn on it.
    """
    fu = cam_param["affine_intrinsics_matrix"][0][0]
    fv = cam_param["affine_intrinsics_matrix"][1][1]
    cu = cam_param["affine_intrinsics_matrix"][0][2]
    cv = cam_param["affine_intrinsics_matrix"][1][2]
    affine_intrinsics_matrix = np.array(cam_param["affine_intrinsics_matrix"])

    rot_mat = np.array(cam_param["extrinsic_matrix"])
    rot_mat[1:, :] *= -1

    camera_position = np.array(cam_param["xyz"])
    #divided by 1000 to convert to meters
    camera_position = camera_position/1000 if adj_cam else camera_position

    distortion = np.array(cam_param["distortion"])
    
    for i in range(len(keypoint_data)):
        keypoints = keypoint_data[i]

        # World to cam
        translated = keypoints[0:3] - camera_position
        kpts_camera = (rot_mat @ translated.T).T

        # Cam to pixel
        Xc = kpts_camera[0]
        Yc = kpts_camera[1]
        Zc = kpts_camera[2]
        u = fu * (Xc / Zc) + cu
        v = fv * (Yc / Zc) + cv
        uv = np.stack([u, v], axis=-1)

        if undistort:
            # Undistort the points
            uv = cv2.undistortPoints(np.expand_dims(uv, axis=0), affine_intrinsics_matrix, distortion, None, affine_intrinsics_matrix)

            # Convert from normalized coordinates to pixel coordinates
            uv = uv.squeeze(axis=0)
            u, v = uv[0][0], uv[0][1]

        if rotate:
            #rotate the points
            u,v = v,u
            #flip the y
            v = frame.shape[0] - v

        #check if u and v are is nan
        if np.isnan(u) or np.isnan(v):
            continue

        # Draw the keypoints on the frame
        print(u,v)
        frame = cv2.circle(frame, (int(u), int(v)), 5, get_color(i), -1)
        
        # Add text at the keypoints
        frame = cv2.putText(frame, str(i), (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame

def get_color(i):
    """
    Generate a unique color based on the input number. (Same color for the same number)

    Args:
        i (int): The input number to generate a color for.

    Returns:
        tuple: A tuple representing the color in BGR format (blue, green, red).
    """
    hash_value = int(hashlib.md5(str(i).encode()).hexdigest(), 16)  # Hash the number
    r = (hash_value % 256)  
    g = ((hash_value // 256) % 256)
    b = ((hash_value // 256 // 256) % 256)
    return (r, g, b)  # OpenCV uses BGR format

def get_project_2d(keypoint_data, cam_param, undistort=False, adj_cam=False,rotate=False,img_height=None):
    """
    Projects 3D keypoints onto a 2D image plane based on camera parameters.

    Args:
        keypoint_data (list): A list of 3D keypoints (N x 3 array) to be projected.
        cam_param (dict): Camera parameters, including intrinsic, extrinsic matrices, and distortion coefficients.
        undistort (bool): Flag indicating whether to undistort the points. Default is False.
        adj_cam (bool): Flag to adjust the camera position (scale it to meters). Default is False.
        rotate (bool): Flag to rotate the points. Default is False.
        img_height (int): The height of the image. Required if rotate is True.

    Returns:
        ndarray: The frame with the projected keypoints drawn on it.
    """
    fu = cam_param["affine_intrinsics_matrix"][0][0]
    fv = cam_param["affine_intrinsics_matrix"][1][1]
    cu = cam_param["affine_intrinsics_matrix"][0][2]
    cv = cam_param["affine_intrinsics_matrix"][1][2]
    affine_intrinsics_matrix = np.array(cam_param["affine_intrinsics_matrix"])

    rot_mat = np.array(cam_param["extrinsic_matrix"])
    rot_mat[1:, :] *= -1

    camera_position = np.array(cam_param["xyz"])
    #divided by 1000 to convert to meters
    camera_position = camera_position/1000 if adj_cam else camera_position

    distortion = np.array(cam_param["distortion"])
    
    #create an array to store the projected keypoints
    projected_keypoints = np.zeros((len(keypoint_data),2))
    for i in range(len(keypoint_data)):
        keypoints = keypoint_data[i]

        # World to cam
        translated = keypoints[0:3] - camera_position
        kpts_camera = (rot_mat @ translated.T).T

        # Cam to pixel
        Xc = kpts_camera[0]
        Yc = kpts_camera[1]
        Zc = kpts_camera[2]
        u = fu * (Xc / Zc) + cu
        v = fv * (Yc / Zc) + cv
        uv = np.stack([u, v], axis=-1)

        if undistort:
            # Undistort the points
            uv = cv2.undistortPoints(np.expand_dims(uv, axis=0), affine_intrinsics_matrix, distortion, None, affine_intrinsics_matrix)

            # Convert from normalized coordinates to pixel coordinates
            uv = uv.squeeze(axis=0)
            u, v = uv[0][0], uv[0][1]
        
        if rotate:
            if img_height is None:
                print("Error: Image height is required for rotating the points")
                return None
            
            #rotate the points
            u,v = v,u
            #flip the y
            v = img_height - v
        
        #store the projected keypoints
        projected_keypoints[i] = [u,v]
    return projected_keypoints

if __name__ == "__main__":
    import os
    vid_path = "./AthletePose3D/data/train_set/S1/Axel_1_cam_1.mp4"
    npy_file = "./AthletePose3D/data/train_set/S1/Axel_1_cam_1.npy"
    json_path = "./AthletePose3D/data/train_set/S1/Axel_1_cam_1.json"
    cam_param_path = "./AthletePose3D/cam_param.json"
    rig_path = os.getcwd()+"/rig.json"

    # Get the motion captured keypoints
    np_keypoint_data = np.load(npy_file)
    keypoint_data = np_keypoint_data[0]
    
    # Map the keypoint data to pose format
    # Pose mapping option "Human3.6M_RM" or "COCO_RM" for running for the rest "Human3.6M" or "COCO"
    joint_names, marker_idxs = mp.load_rig_mapping(rig_path, "Human3.6M", get_marker_labels(json_path)) 
    keypoint_data = mp.apply_rig_format(keypoint_data, joint_names, marker_idxs)

    # Get the camera parameters
    cam_param = get_cam_param(cam_param_path, json_path)

    # Get the first frame of the video as an example
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()

    # Project the 2D keypoints
    projected_keypoints = get_project_2d(keypoint_data, cam_param)
    # Project the 2D keypoints, the dataset option are "Human3.6M" or "COCO"
    frame_2d = vis.show2Dpose(projected_keypoints, frame, unique_color=False, dataset="Human3.6M")

    # Save the frame
    cv2.imwrite("example.png", frame_2d)

    # Close the video
    cap.release()
