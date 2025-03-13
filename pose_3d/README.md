# Training or Fine-Tuning a 3D Pose Estimation Model with AthletePose3D  

This guide explains how to train or fine-tune a **3D pose estimation model** using the **AthletePose3D** dataset.  

## Optional: Create Training Data  
Before training, you can generate and convert the dataset using the following commands:
(Already done for AthletePose3D and located in `AthletePose3D/pose_3d`)  

```bash
python gendb.py
python convert.py
```
## Updating the Data Reader
To use the AthletePose3D dataset, update the DataReaderH36M class in the data reader, example with [TCPFormer](https://github.com/AsukaCamellia/TCPFormer):

1. Locate the file:
```
./TCPFormer/data/reader/h36m.py
```
2. Replace the DataReaderH36M class with the one provided in:
```
datareader_ap3d.py
```