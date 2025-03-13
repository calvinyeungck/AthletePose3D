# Training or Fine-Tuning a 2D Pose Estimation Model with AthletePose3D  

To train or fine-tune a 2D pose estimation model using the **AthletePose3D** dataset, follow the [MogaNet](https://github.com/Westlake-AI/MogaNet) example below.  

## Detailed Instructions  
For exact setup and customization details, refer to the official **MMPose documentation**:  
🔗 [Customize Datasets in MMPose](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_datasets.html)  

## Dataset Location  
The dataset is located at:  `AthletePose3D/pose_2d`
### **Provided Files**  

#### **1. Annotations (COCO Format)**  
- JSON files in **COCO format** are stored in:  `AthletePose3D/pose_2d/annotations`
- The **`valid_set_100`** can be used for testing before running the full dataset.  

#### **2. Dataset Info Configuration**  
- The dataset info config file is located in the repo:  `./pose_2d/ap2d.py`

#### **3. Top-Down Dataset Class**  
- The top-down dataset class is available in the repo:  `./pose_2d/topdown_ap2d.py`

#### **4. Training and Testing Configurations**
- The training and testing configurations are stored in the repo:  `./pose_2d/moganet_b_ap2d_384x288.py`
