# 🏃 AthletePose3D: A Benchmark Dataset for 3D Human Pose Estimation and Kinematic Validation in Athletic Movements (CVSports at CVPR 2025)

<p align="center">
  <a href="https://arxiv.org/abs/2503.07499">
    <img src="https://img.shields.io/badge/ArXiv-2503.07499-b31b1b?style=for-the-badge&logo=arxiv" alt="ArXiv">
  </a>
  <a href="https://github.com/calvinyeungck/AthletePose3D/tree/main/license">
    <img src="https://img.shields.io/badge/Download-AthletePose3D-blue?style=for-the-badge&logo=databricks" alt="Download">
  </a>
</p>


## 📌 Overview  
**AthletePose3D** (AP3D) is a novel dataset for **monocular 3D human pose estimation** in **sports biomechanics**, designed to capture **high-speed, high-acceleration movements**. Alongside the raw dataset, we also provide a **training-ready version** prepared for **2D and 3D pose estimation modeling**, including both preprocessed annotations and AP3D fine-tuned model parameters.

> **⚠️ Important Notes:**  
> **11/07/2025 – Erratum:** A preprocessing mistake occurred in one camera angle of the running motions (3D). The `pose_3d.zip` files, pre-trained model, and corresponding code have been corrected — please re-download them. For the updated results, please refer to the arXiv paper v3. The conclusions of the experiment and the overall paper remain unchanged.  
> **21/06/2025 – Update:** The `pose_3d.zip` files and code have been corrected. Please re-download them. Thank you.

**To download the dataset or fine-tuned checkpoints, please read the [license agreement](https://github.com/calvinyeungck/AthletePose3D/tree/main/license).**

<p align="center">
  <img src="https://github.com/calvinyeungck/AthletePose3D/blob/main/fig/cvsports2025.png" alt="alt text">
</p>

## 📂 Dataset Features  
- 🏅 **12 sports motions** across various disciplines  
- 🎞️ **1.3M frames** & **165K postures**  
- ⚡ Focus on **high-intensity athletic movements**  

## 📊 Model Evaluation  
- 📉 **SOTA models trained on conventional datasets struggle with athletic motions**  
- 🎯 Fine-tuning on **AthletePose3D** reduces **MPJPE from 214mm → 65mm** (**69% improvement!**)  

## 🔬 Kinematic Validation  
- ✅ Strong **joint angle correlation**  
- ⚠️ **Limitations in velocity estimation**  

## 🚀 Contribution  
- Benchmarking **monocular pose estimation for sports**  
- Advancing **pose estimation in high-performance environments**  

## 💡 Example 
<div><video controls src="https://github.com/user-attachments/assets/3efcb8ab-2ede-4463-ad62-c9544ed54d40" muted="false"></video></div>

## 📂 Dataset Structure

- `/AthletePose3D/`
  - `/data/`                      (video and motion data)
    - `/train_set/`
      - `/S1/`                      (subject)
        - `Axel_1_cam_1.mp4`      (video file)
        - `Axel_1_cam_1.json`     (video and motion information)
        - `Axel_1_cam_1.npy`      (motion data)
        - `Axel_1_cam_1_coco.npy` (COCO keypoints)
        - `Axel_1_cam_1_h36m.npy` (H3.6M keypoints)
      - `/S2/`
      - ...
    - `/valid_set/`
    - `/test_set/`
  - `/pose_2d/`                   (2D pose estimation ready data)
    - `/annotations`/               (Annotations in COCO Format)
      - `train_set.json`
      - ...
    - `/det_result/`                (Detected with YOLOv8)
      - `ap2d_train_det.json`
      - ...
    - `/train_set/`                 (Image files)          
    - `/valid_set/`
    - `/test_set/`
  - `/pose_3d/`                   (3D pose estimation ready data)   
    - `/frame_81/`                
    - `train.pkl`
    - `valid.pkl`
  - `cam_param.json`              (camera parameters)


## ⬇️ Download AthletePose3D
The dataset is available for download at the following link: [Download AthletePose3D](https://github.com/calvinyeungck/AthletePose3D/tree/main/license)

## 📖 Reference
Please consider citing our work if you find it helpful to yours:

```
@misc{yeung2025athletepose3d,
      title={AthletePose3D: A Benchmark Dataset for 3D Human Pose Estimation and Kinematic Validation in Athletic Movements}, 
      author={Calvin Yeung and Tomohiro Suzuki and Ryota Tanaka and Zhuoer Yin and Keisuke Fujii},
      year={2025},
      eprint={2503.07499},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.07499}, 
}
```

## 📄 License
For non-commercial and scientific research purposes only. For details refer to [LICENSE](https://github.com/calvinyeungck/AthletePose3D/tree/main/license)
