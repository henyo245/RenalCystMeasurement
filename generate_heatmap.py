import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from numpy import linalg as LA
import skimage.io as io
#from train_test_split import split_not_random
import os
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

matrix=cv2.imread('matrix_crop.png', cv2.IMREAD_GRAYSCALE)

def my_makedirs(path):
    if not os.path.exists(path):
         os.makedirs(path)
            
# pre processing
def gaussian_generator(x,y,r,bool_normalize):
    x_axis=cv2.getGaussianKernel(x,r)
    y_axis=cv2.getGaussianKernel(y,r)
    
    matrix=x_axis*y_axis.T
    
    if bool_normalize==True:
        max_val=255
    else:
        max_val=1
        
    matrix=max_val*(matrix/np.max(matrix))
    matrix=np.array(matrix,np.uint8)
    
    #plt.imshow(matrix)
    
    return matrix

def crop(img, true_coord, imgsz):
    coord0 = true_coord[0]
    coord1 = true_coord[1]
    c = (np.array(coord0)+np.array(coord1))/2
    r = np.linalg.norm(np.array(coord0)-np.array(coord1))/2

    pt1 = np.array(c)-[r,r]
    pt2 = np.array(c)+[r,r]
    
    bbox_size = pt2[1]-pt1[1]
    margin = int(bbox_size*0.2)

    crop_img = img[int(pt1[1])-margin:int(pt2[1])+margin,int(pt1[0])-margin:int(pt2[0])+margin]
    if crop_img.shape[0]==0 or crop_img.shape[1]==0:
        return 0
    else:
        coord0_crop = [coord0[0]-pt1[0]+margin,coord0[1]-pt1[1]+margin]
        coord1_crop = [coord1[0]-pt1[0]+margin,coord1[1]-pt1[1]+margin]
        crop_img = cv2.resize(crop_img, dsize=(imgsz, imgsz))
        coord0_crop = list(map(lambda x: round(x * imgsz/(bbox_size+margin*2)), coord0_crop))
        coord1_crop = list(map(lambda x: round(x * imgsz/(bbox_size+margin*2)), coord1_crop))
        coord0_crop = list(map(lambda x: int(x) , coord0_crop))
        coord1_crop = list(map(lambda x: int(x) , coord1_crop))

        return crop_img, (coord0_crop,coord1_crop), (pt1,pt2)

def heatmap(coords, imgsz):
    x1,y1 = int(coords[0][0]),int(coords[0][1])
    x2,y2 = int(coords[1][0]),int(coords[1][1])
    
    dm=np.zeros((imgsz, imgsz, 3), np.uint8)

    dm1=matrix[imgsz-y1:imgsz-y1+imgsz, imgsz-x1:imgsz-x1+imgsz]
    dm2=matrix[imgsz-y2:imgsz-y2+imgsz, imgsz-x2:imgsz-x2+imgsz]

    dm[:,:,0]=dm1
    dm[:,:,1]=dm2
    #dm[:,:,2]=255-dm1-dm2
    
    return dm

def ext_yellow(image):
    bgrLowernp.array([255,255,0])
    bgrUpper=np.array([255,255,0])
    img_mask=cv2.inRange(image,bgrLower,bgrUpper)
    result=cv2.bitwise_and(image, image, mask=img_mask)
    return result

def generate_input(source, save_dir, df_path, imgsz):
    #save_dir = os.path.jpin(source, 'mask_input')
    #save_dir = source
    my_makedirs(save_dir)
    # Make directories named each label
    new_dir = os.path.join(save_dir, 'groundTruth')
    my_makedirs(new_dir)
    new_dir = os.path.join(save_dir, 'input')
    my_makedirs(new_dir)
    
    #GRAY_DIR='/data1/kanauchi/US_Capture/mask_input'
    files=sorted(os.listdir(source))
    #with open('/data1/kanauchi/US_Capture/coords_only2.pickle', 'rb') as web:
    with open(df_path, 'rb') as web:
        coords = pickle.load(web)

    #files = os.listdir(save_dir)
    for file in tqdm(files):
        if file[-3:] == 'png':
            img = plt.imread(os.path.join(source, file))
            coord = coords[file]

            if crop(img, coord, imgsz)!=0:
                crop_img, (coord0_crop,coord1_crop), (pt1,pt2) = crop(img, coord, imgsz)
                #input
                cv2.imwrite(os.path.join(save_dir, 'input', file[:-4]+'.png'), crop_img*255)

                #ground truth
                dm = heatmap([coord0_crop,coord1_crop], imgsz)
                cv2.imwrite(os.path.join(save_dir, 'groundTruth', file[:-4]+'.png'),dm)
        

def generate_images(source):
    #generate_input(source)
    #source='/tmp/data/newdatas/dm_2points'
    split_not_random(source)

def run(source=ROOT / 'original_data',  # dir
        save_dir = ROOT / 'data',  # dir
        df_path =  ROOT / 'coordinate.pickle', 
        imgsz=256,
        ):
    
    generate_input(source, save_dir, df_path, imgsz)
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'original_data', help='original data dir')
    parser.add_argument('--save_dir', type=str, default=ROOT / 'data', help='save dir')
    parser.add_argument('--df_path', type=str, default=ROOT / 'coordinate.pickle', help='df path')
    parser.add_argument('--imgsz', '--img', '--img-size',  type=int, default=128, help='inference size h,w')
    opt = parser.parse_args()
    #print_args(FILE.stem, opt)
    return opt


def main(opt):
    #check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)