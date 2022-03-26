# Renal Cyst Measurement

This repository contains code to detect and measure renal cysts on abdominal ultrasound images.
<img src="https://github.com/henyo245/RenalCystMeasurement/blob/master/img.png" width=30% height=30%> <img src="https://github.com/henyo245/RenalCystMeasurement/blob/master/img_pred.png" width=30% height=30%>
## Requirement
For prediction and UNet++ training, this project requires 
- cuda8
- python3.6
- numpy
- opencv-python
- matplotlib
- h5py==2.10.0
- Keras==2.2.2
- tensorflow-gpu==1.4.0
#tensorflow-gpu==1.4.1
- scikit-image
- pytorch-lightning
- torch
- torchvision
- pandas
- seaborn

See the YOLOv5 repository for YOLOv5 training requirements.

## Examples
```
python3.6 prediction.py --source img.png
```

## Reference
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [UNet++](https://github.com/MrGiovanni/UNetPlusPlus)
