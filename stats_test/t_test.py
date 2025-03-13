import pickle
import pdb
import numpy as np
import spm1d
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

def resample_signal(signal, target_length):
    """Resamples a 1D signal to the target length using linear interpolation.
       If NaNs are present, they are replaced with the previous valid value.
       Each sample is min-max normalized before interpolation.
    """
    # Replace NaNs with the previous valid value
    mask = np.isnan(signal)
    if np.any(mask):  # Only process if there are NaNs
        for i in range(1, len(signal)):  # Start from index 1 to avoid out-of-bounds error
            if mask[i]:  # If NaN, copy the previous valid value
                signal[i] = signal[i - 1]
    
    # Apply Min-Max normalization
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val != min_val:  # Avoid division by zero if all values are the same
        signal = (signal - min_val) / (max_val - min_val)
    
    # If original length is the same as the target length, no interpolation is needed
    if len(signal) == target_length:
        return signal
    
    # Interpolation
    x_original = np.linspace(0, 1, len(signal))  # Original time points
    x_target = np.linspace(0, 1, target_length)  # Target time points
    interpolator = interp1d(x_original, signal, kind='linear')

    # Check if there is any NaN after interpolation
    resampled_signal = interpolator(x_target)
    if np.isnan(resampled_signal).any():
        raise ValueError("NaN values detected after interpolation.")

    return resampled_signal

def drop_nan_arrays(list_of_methods):
    """Drops arrays with all NaN values from each method in the list."""
    
    # Iterate over each method's list of arrays
    drop_list = []
    for i in range(len(list_of_methods[0])):
        for j in range(4):
            if np.isnan(list_of_methods[0][i][j]).all():
                drop_list.append(i)
            if np.isnan(list_of_methods[1][i][j]).all():
                drop_list.append(i)
            if np.isnan(list_of_methods[2][i][j]).all():
                drop_list.append(i)
            if np.isnan(list_of_methods[3][i][j]).all():
                drop_list.append(i)
    drop_list = np.unique(drop_list)
    for i in sorted(drop_list, reverse=True):
        list_of_methods[0].pop(i)
        list_of_methods[1].pop(i)
        list_of_methods[2].pop(i)
        list_of_methods[3].pop(i)
    return list_of_methods

pickle_path = './test_set_results/kinematics/pose_velocity.pkl'
save_path = './test_set_results/kinematics/'
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

velocity = [data['joint_pose_2d_vel_list'], data['joint_pose_img_vel_list'], data['joint_gt_2d_vel_list'], data['joint_gt_3d_world_vel_list']]
angle = [data['angle_2d_list'],data['angle_img_list'],  data['angle_gt_2d_list'], data['angle_gt_3d_world_list']]

#get the max length
max_length = 0
for i in range(len(velocity)):
    for j in range(len(velocity[i])):
        if velocity[i][j].shape[1] > max_length:
            max_length = velocity[i][j].shape[1]

#interpolate to the max length
new_velocity = []
for i in range(len(velocity)):
    new_velocity.append([])
    for j in range(len(velocity[i])):
        new_velocity[i].append([])
        for k in range(velocity[i][j].shape[0]):
            new_velocity[i][j].append(resample_signal(velocity[i][j][k], max_length))

new_velocity = np.array(new_velocity)

angle_drop = drop_nan_arrays(angle)
angle = angle_drop.copy()

new_angle = []
for i in range(len(angle)):
    new_angle.append([])
    for j in range(len(angle[i])):
        new_angle[i].append([])
        for k in range(angle[i][j].shape[0]):
            try:
                new_angle[i][j].append(resample_signal(angle[i][j][k], max_length))
            except ValueError:
                pdb.set_trace()
            # new_angle[i][j].append(resample_signal(angle[i][j][k], max_length))

new_angle = np.array(new_angle)

#get 2d estimate and 3d estimate, and 3d gt
#arms (11,12,13,14,15,16)
arms_vel = new_velocity[:, :, [11,12,13,14,15,16], :]
#reshape to 3d array
arms_vel = arms_vel.reshape(arms_vel.shape[0], -1, arms_vel.shape[3])
arms_2d = arms_vel[0]
arms_3d = arms_vel[1]
arms_gt = arms_vel[3]

#feets (1,2,3,4,5,6)
feets_vel = new_velocity[:, :, [1,2,3,4,5,6], :]
#reshape to 3d array
feets_vel = feets_vel.reshape(feets_vel.shape[0], -1, feets_vel.shape[3])
feets_2d = feets_vel[0]
feets_3d = feets_vel[1]
feets_gt = feets_vel[3]

#for angle
arms_angle = new_angle[:, :, [2,3], :]
#reshape to 3d array
arms_angle = arms_angle.reshape(arms_angle.shape[0], -1, arms_angle.shape[3])
arms_2d_angle = arms_angle[0]
arms_3d_angle = arms_angle[1]
arms_gt_angle = arms_angle[3]

feets_angle = new_angle[:, :, [1,0], :]
#reshape to 3d array
feets_angle = feets_angle.reshape(feets_angle.shape[0], -1, feets_angle.shape[3])
feets_2d_angle = feets_angle[0]
feets_3d_angle = feets_angle[1]
feets_gt_angle = feets_angle[3]

#check if there is nan

# # Compute SPM paired t-test 2d vs gt
t_arm = spm1d.stats.ttest_paired(arms_2d, arms_gt)
ti_arm = t_arm.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(arms_2d.flatten(), arms_gt.flatten())
print(f'Correlation between 2D and GT for arms: {corr}')
print(ti_arm)

t_feet = spm1d.stats.ttest_paired(feets_2d, feets_gt)
ti_feet = t_feet.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(feets_2d.flatten(), feets_gt.flatten())
print(f'Correlation between 2D and GT for feets: {corr}')
print(ti_feet)

# # Compute SPM paired t-test 3d vs gt
t_arm = spm1d.stats.ttest_paired(arms_3d, arms_gt)
ti_arm = t_arm.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(arms_3d.flatten(), arms_gt.flatten())
print(f'Correlation between 3D and GT for arms: {corr}')
print(ti_arm)

t_feet = spm1d.stats.ttest_paired(feets_3d, feets_gt)
ti_feet = t_feet.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(feets_3d.flatten(), feets_gt.flatten())
print(f'Correlation between 3D and GT for feets: {corr}')
print(ti_feet)

# # Compute SPM paired t-test 2d vs 3d
t_arm = spm1d.stats.ttest_paired(arms_2d, arms_3d)
ti_arm = t_arm.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(arms_2d.flatten(), arms_3d.flatten())
print(f'Correlation between 2D and 3D for arms: {corr}')
print(ti_arm)

t_feet = spm1d.stats.ttest_paired(feets_2d, feets_3d)
ti_feet = t_feet.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(feets_2d.flatten(), feets_3d.flatten())
print(f'Correlation between 2D and 3D for feets: {corr}')
print(ti_feet)



# Compute SPM paired t-test 2d vs gt
t_arm = spm1d.stats.ttest_paired(arms_2d_angle, arms_gt_angle)
ti_arm = t_arm.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(arms_2d_angle.flatten(), arms_gt_angle.flatten())
print(f'Correlation between 2D and GT for arms angle: {corr}')
print(ti_arm)

t_feet = spm1d.stats.ttest_paired(feets_2d_angle, feets_gt_angle)
ti_feet = t_feet.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(feets_2d_angle.flatten(), feets_gt_angle.flatten())
print(f'Correlation between 2D and GT for feets angle: {corr}')
print(ti_feet)

# Compute SPM paired t-test 3d vs gt
t_arm = spm1d.stats.ttest_paired(arms_3d_angle, arms_gt_angle)
ti_arm = t_arm.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(arms_3d_angle.flatten(), arms_gt_angle.flatten())
print(f'Correlation between 3D and GT for arms angle: {corr}')
print(ti_arm)

t_feet = spm1d.stats.ttest_paired(feets_3d_angle, feets_gt_angle)
ti_feet = t_feet.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(feets_3d_angle.flatten(), feets_gt_angle.flatten())
print(f'Correlation between 3D and GT for feets angle: {corr}')
print(ti_feet)

# Compute SPM paired t-test 2d vs 3d
t_arm = spm1d.stats.ttest_paired(arms_2d_angle, arms_3d_angle)
ti_arm = t_arm.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(arms_2d_angle.flatten(), arms_3d_angle.flatten())
print(f'Correlation between 2D and 3D for arms angle: {corr}')
print(ti_arm)

t_feet = spm1d.stats.ttest_paired(feets_2d_angle, feets_3d_angle)
ti_feet = t_feet.inference(alpha=0.05, two_tailed=True)  # 5% significance level
corr, _ = pearsonr(feets_2d_angle.flatten(), feets_3d_angle.flatten())
print(f'Correlation between 2D and 3D for feets angle: {corr}')
print(ti_feet)


