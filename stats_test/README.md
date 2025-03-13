# Performing a Paired t-Test statistical parametric mapping (SPM) on Kinematics Data  

Follow these steps to conduct a paired t-test on kinematics data.  

## Step 1: Install Required Packages  
Ensure you have the necessary dependencies installed:  

```bash
pip3 install scipy numpy
```

## Step 2: Run Pose Estimation Models and Save Results
Run the 2D and 3D pose estimation models and save the outputs using the following code:
```python
np.savez_compressed(out_path_2d, reconstruction=pose_2d)  # Save 2D pose
np.savez_compressed(out_path_3d, reconstruction=pose_3d)  # Save 3D pose
```
## Step 3: Compute Kinematics and Perform t-Test
Execute the following commands to process kinematics data and run the t-test:
```python
python get_kinematics.py
python t_test.py
```