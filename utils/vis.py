#ref: https://github.com/TaatiTeam/MotionAGFormer
import cv2
import numpy as np
import hashlib
import pdb
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def show2Dpose(kps, image, unique_color=False, dataset="Human3.6M"):
    if "Human3.6M" in dataset:
        return show2Dpose_h36m(kps, image, unique_color)
    elif "COCO" in dataset:
        return show2Dpose_coco(kps, image, unique_color)
    else:
        raise ValueError(f"Dataset {dataset} not supported. Please use 'h36m' or 'coco'.")


def show2Dpose_h36m(kps, image, unique_color=False):
    img = image.copy()
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        if unique_color == True:
            color = get_color(j)
            cv2.circle(img, (start[0], start[1]), thickness=-1, color=color, radius=3)
            cv2.circle(img, (end[0], end[1]), thickness=-1, color=color, radius=3)
        else:
            cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
            cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show2Dpose_coco(kps, image, unique_color=False):
    img = image.copy()
    connections = [[0, 6], [0, 5], [6, 5], [6, 8], [8, 10],
                   [5, 7], [7, 9], [6, 12], [5, 11], [12, 11],
                   [12, 14], [14, 16], [11, 13], [13, 15]]

    LR = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        if unique_color == True:
            color = get_color(j)
            cv2.circle(img, (start[0], start[1]), thickness=-1, color=color, radius=3)
            cv2.circle(img, (end[0], end[1]), thickness=-1, color=color, radius=3)
        else:
            cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
            cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img

def show3Dpose(vals,save_path=None,scale=False):
    fig = plt.figure(figsize=(9.6, 5.4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05) 
    ax = plt.subplot(gs[0], projection='3d')
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    #divide all value in vals by 1000
    if scale:
        vals = vals/1000

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    # Remove or adjust tick visibility settings
    ax.tick_params('x', labelbottom=True)
    ax.tick_params('y', labelleft=True)
    ax.tick_params('z', labelleft=True)

    # Set ticks and labels for the axes
    ax.set_xticks(np.arange(-RADIUS + xroot, RADIUS + xroot, step=1))  # Set X ticks
    ax.set_yticks(np.arange(-RADIUS + yroot, RADIUS + yroot, step=1))  # Set Y ticks
    ax.set_zticks(np.arange(-RADIUS_Z + zroot, RADIUS_Z + zroot, step=1))  # Set Z ticks

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory
        print(f"Figure saved to {save_path}")
    
    return fig if not save_path else None



def get_color(i, total_colors=17):
    """
    Generate a unique, visually distinct color for a given index.

    Args:
        i (int): The index for which to generate a color.
        total_colors (int): The total number of distinct colors needed.

    Returns:
        tuple: A tuple representing the color in BGR format (blue, green, red).
    """
    hue = int(180 * (i / total_colors))  # Spread across HSV hue range
    saturation = 200  # Keep colors vibrant
    value = 255  # Max brightness

    # Convert HSV to BGR
    color_bgr = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color_bgr)  # Ensure tuple format

