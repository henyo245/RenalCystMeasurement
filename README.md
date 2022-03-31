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

## Data
```
RenalCystMeasurement/data
├unetpp
│ ├input
│ │ ├image1.png
│ │ ├image2.png
│ │ ├...
│ │
│ └groundTruth
│ │ ├image1.png
│ │ ├image2.png
│ │ ├...
│ 
└yolov5
  ├images
  │ ├image1.png
  │ ├image2.png
  │ ├...
  │
  └labels
    ├image1.txt
    ├image2.txt
    ├...
```

### Format of coordinate.pickle
```
{'image1.png': ((x1, y1),(x2, y2)),
 'image2.png': [((x1, y1),(x2, y2)),
                ((x3, y3),(x4, y4)),
                ((x5, y5),(x6, y6))],
 ...
 }
```

## Examples
```
# prediction
python3.6 prediction.py --source img.png

# train YOLOv5
cd yolov5
python3.6 train.py --epoch 100 --data cyst_multiclass.yaml --weights yolov5m.pt --img 256 --batch 16

# generate heatmap for UNet++
python3.6 generate_heatmap.py

# train UNet++
python3.6 train_unetpp.py
```

## Reference
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [UNet++](https://github.com/MrGiovanni/UNetPlusPlus)
