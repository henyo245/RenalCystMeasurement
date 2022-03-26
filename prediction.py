import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
import gc
import pickle
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from IPython.display import Image, clear_output  # to display images

sys.path.append('yolov5')  # add yolov5/ to path
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from unetprusprus_2points import *
from result_multi import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

##--------Measurement Marker Coordinate Prediction---------##
def return_coordinate(imgsz, coord_crop, pt):
    #IMG_SIZE = 256

    coord0_crop = coord_crop[0]
    coord1_crop = coord_crop[1]
    pt1 = pt[0]
    pt2 = pt[1]
    
    if pt2[1]-pt1[1]>pt2[0]-pt1[0]:
        bbox_size = pt2[1]-pt1[1]
    else:
        bbox_size = pt2[0]-pt1[0]
    
    margin = int(bbox_size*0.2)
    
    coord0_re = list(map(lambda x: x / imgsz*(bbox_size+margin*2), coord0_crop))
    coord1_re = list(map(lambda x: x / imgsz*(bbox_size+margin*2), coord1_crop))
    coord0_re = [coord0_re[1]+pt1[0]-margin,coord0_re[0]+pt1[1]-margin]
    coord1_re = [coord1_re[1]+pt1[0]-margin,coord1_re[0]+pt1[1]-margin]
    coord0_re = list(map(lambda x: round(x), coord0_re))
    coord1_re = list(map(lambda x: round(x), coord1_re))
    
    return (coord0_re,coord1_re)

def direction_img(dm,tmp_zahyo):
    width=dm.shape[1]
    height=dm.shape[0]
    center=(height/2, width/2)
    if dm[tmp_zahyo[0][0],tmp_zahyo[0][1],0]<dm[tmp_zahyo[1][0],tmp_zahyo[1][1],1]:
        return 1
    else:
        return 0
    
def pred2coord(imgsz, img_path, Y_pred_denorm, pts):
    pred_coords={}
    file = os.path.basename(img_path)
    key = file
    for dm,pt in zip(Y_pred_denorm,pts):   
        #plt.imshow(dm);plt.show()       
        width = dm.shape[1]
        #print(MASK_DIR + file_names[i])
        tmp_coord = []
        for i in range(2):
            a=np.argmin(dm[:,:,i])
            q, mod=divmod(a, width)
            #print((mod,q), coord[i])
            tmp_coord.append((int(q),int(mod)))
        channel = direction_img(dm,tmp_coord)
        
        tmp_coord_new = return_coordinate(imgsz, tmp_coord, pt)
        if abs((pt[1][0]-pt[0][0])-np.linalg.norm(np.array(tmp_coord_new[0])-np.array(tmp_coord_new[1]))) > (pt[1][0]-pt[0][0])*0.05:
            print(tmp_coord)
            print('correction:',(pt[1][0]-pt[0][0]),np.linalg.norm(np.array(tmp_coord_new[0])-np.array(tmp_coord_new[1])))
            tmp_coord[channel] = (width-tmp_coord[1-channel][0],width-tmp_coord[1-channel][1])
        tmp_coord = return_coordinate(imgsz, tmp_coord, pt)
        if file in pred_coords:
            pred_coords[file] = pred_coords[file]+(tuple(tmp_coord))
        else:
            pred_coords[file] = (tuple(tmp_coord))         
    
    return pred_coords
    
def crop(img_detections, img_path):
    IMAGE_SIZE = 256
    images = []
    pts = []
    
    img = plt.imread(img_path)
    #plt.imshow(img);plt.show()
    height, width = img.shape[0], img.shape[1]
    
    count = 0
    for d in img_detections:       
        #print(d)
        x1, y1, x2, y2 = d[:4]
        pts.append([(x1,y1),(x2,y2)])
        #x1, y1, x2, y2 = int(x1),int(x2),int(y1),int(y2)
        #x_center, y_center=(int((x1+y1)/2), int((x2+y2)/2))
        
        if x2-x1>y2-y1:
            bbox_size = x2-x1
        else:
            bbox_size = y2-y1
        margin = int(bbox_size*0.2)
        #print(margin)
        #print(int(y1)-margin,int(y2)+margin,int(x1)-margin,int(x2)+margin)
        crop_img = img[int(y1)-margin:int(y2)+margin,int(x1)-margin:int(x2)+margin]
        crop_img = cv2.resize(crop_img, dsize=(IMAGE_SIZE, IMAGE_SIZE))
        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
        image[:,:,0]=crop_img.copy()
        image[:,:,1]=crop_img.copy()
        image[:,:,2]=crop_img.copy()
        images.append(image)
        
        count+=1
        
    X_test = np.zeros((count, IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)
    for idx,image in enumerate(images):
        X_test[idx] = normalize_x(image*255)
        #plt.imshow(image);plt.show()
    
    return X_test, pts
##----End of Measurement Marker Coordinate Prediction----##

def run(weights_yolov5=ROOT / 'yolov5/best.pt',  # model.pt path(s)
        weights_unetpp=ROOT / 'UNetPlusPlus-master/unetplus_weights.hdf5',  # model.hdf5 path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(256, 256),  # inference size (height, width)
        conf_thres=0.1,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    ##---------------------Renal Cysts Detection-----------------------##
    clear_output()
    print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
    
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    detect_model = attempt_load(weights_yolov5, map_location=device)  # load FP32 model
    stride = int(detect_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz[0], s=stride)  # check image size
    names = detect_model.module.names if hasattr(detect_model, 'module') else detect_model.names  # get class names

    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    matrix=cv2.imread('matrix.bmp', cv2.IMREAD_GRAYSCALE)
    
    # Run inference
    if device.type != 'cpu':
        detect_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(detect_model.parameters())))  # run once
    t0 = time.time()
    for source, img, im0s, vid_cap in dataset:
        print(source)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = detect_model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = source, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path

            #s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        
    img_detections = det[:, :4].tolist()
    
    print(len(img_detections), 'renal cyst(s) detected')
    ##------------------End of Renal Cysts Detection-----------------##
    ##--------Measurement Marker Coordinate Prediction---------##
    if len(img_detections)==0:
        print('No renal cysts were detected.')
        img = cv2.imread(source)
        #plt.imshow(img);plt.show()
        return
    else:
        print('Loading UNet++')

        # input heatmap
        input_channel_count = 3
        output_channel_count = 1
        first_layer_filter_count = 64
        xnet_model = Xnet(backbone_name='densenet121', encoder_weights=None, decoder_block_type='transpose',decoder_filters=(128,64,32,16,8), classes=3) # build UNet++
        xnet_model.load_weights(weights_unetpp)
        BATCH_SIZE = 20
    
        X_test, pts = crop(img_detections, source)
        Y_pred = xnet_model.predict(X_test, BATCH_SIZE)
       
    img = cv2.imread(source)
    height, width = img.shape[0], img.shape[1]    
    Y_pred_denorm = []

    for y in Y_pred:
        Y_pred_denorm.append(denormalize_y(y))
        
    pred_coords_multi = pred2coord(imgsz, source, Y_pred_denorm, pts)
    ##----End of Measurement Marker Coordinate Prediction----##
    #for img in X_test:
        #plt.imshow(img);plt.show()
    
    for file, pred_coord in pred_coords_multi.items():
        img = cv2.imread(source)
        for idx in range(len(pts)):
            bbox = cv2.rectangle(img, tuple([int(s) for s in pts[idx][0]]), tuple([int(s) for s in pts[idx][1]]), (0, 255, 255), thickness=2)
        img = write_marker(img, pred_coord, 1)
        #plt.imshow(img);plt.show()
        save_path = source[:-4]+'_pred'+source[-4:]
        cv2.imwrite(save_path, img)
        print('saved to ', save_path)
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_yolov5', nargs='+', type=str, default=ROOT / 'yolov5/best.pt', help='YOLOv5 model path(s)')
    parser.add_argument('--weights_unetpp', nargs='+', type=str, default=ROOT / 'UNetPlusPlus-master/unetplus_weights.hdf5', help='UNet++ model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[256], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(FILE.stem, opt)
    return opt


def main(opt):
    #check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)