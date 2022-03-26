import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import shutil
import argparse
from pathlib import Path

sys.path.append('UNetPlusPlus-master')
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from segmentation_models import Unet, Nestnet, Xnet

#from generate_dm_2points import generate_images
#from unetprusprus_2points import train_test
#sys.path.append("../../functions")
#from result_post import result

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Function to normalize a value from -1 to 1
def normalize_x(image):
    image = image/127.5 - 1
    return image


# Function to normalize a value from 0 to 1
def normalize_y(image):
    image = image/255
    return image


# Function to return a value from 0 to 255
def denormalize_y(image):
    image = image*255
    return image


# Function to load input images
def load_X(imgsz, folder_path):
    import os, cv2

    image_files =[f for f in os.listdir(folder_path) if not f[:2] in ['._']]
    image_files.sort()
    images = np.zeros((len(image_files), imgsz, imgsz, 3), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file)
        image = cv2.resize(image, (imgsz, imgsz))
        images[i] = normalize_x(image)
    return images, image_files


# Function to load label image
def load_Y(imgsz, folder_path):
    import os, cv2

    image_files =[f for f in os.listdir(folder_path) if not f[:2] in ['._']]
    image_files.sort()
    images = np.zeros((len(image_files), imgsz, imgsz, 3), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file)
        image = cv2.resize(image, (imgsz, imgsz))
        #image = image[:, :, np.newaxis]
        images[i] = normalize_y(image)
    return images

def predict(X_TEST_DIR, MODEL_WEIGHTS_DIR, SAVE_PREDICT_DIR):
    X_test, file_names = load_X(X_TEST_DIR)

    input_channel_count = 3
    output_channel_count = 1
    first_layer_filter_count = 64
    model = Xnet(backbone_name='densenet121', encoder_weights=None, decoder_block_type='transpose',decoder_filters=(128,64,32,16,8), classes=3) # build UNet++
    model.load_weights(MODEL_WEIGHTS_DIR)
    BATCH_SIZE = 21
    Y_pred = model.predict(X_test, BATCH_SIZE)
    
    # Make directory
    if not os.path.exists(SAVE_PREDICT_DIR):
        os.makedirs(SAVE_PREDICT_DIR)
            
    for i, y in enumerate(Y_pred):
        img = cv2.imread(X_TEST_DIR + os.sep + file_names[i])
        y = cv2.resize(y, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(SAVE_PREDICT_DIR, file_names[i]), denormalize_y(y))

def run(source=ROOT / 'data',  # dir
        imgsz=256,
        epochs = 20,
        batch_size = 16,
        ):
    """
    X_TRAIN_DIR=os.path.join(source, 'images/input_train_test/trainData')
    Y_TRAIN_DIR=os.path.join(source, 'images/groundTruth_train_test/trainData')
    MODEL_WEIGHTS_DIR=os.path.join(source, 'unetpp_weights.hdf5')
    X_TEST_DIR=os.path.join(source, 'imagaes/input_train_test/testData')
    SAVE_PREDICT_DIR=os.path.join(source, 'imageas/prediction')
    """
    
    X_TRAIN_DIR=os.path.join(source, 'input_train_test/trainData')
    Y_TRAIN_DIR=os.path.join(source, 'groundTruth_train_test/trainData')
    MODEL_WEIGHTS_DIR=os.path.join('unetpp_weights.hdf5')
    X_TEST_DIR=os.path.join(source, 'input_train_test/testData')
    SAVE_PREDICT_DIR=os.path.join('prediction')
    
    ##----------------training----------------##
    """
    X_train, file_names = load_X(imgsz, X_TRAIN_DIR)
    Y_train = load_Y(imgsz, Y_TRAIN_DIR)

    # prepare model
    model = Xnet(backbone_name='densenet121', encoder_weights='imagenet', decoder_block_type='transpose',decoder_filters=(128,64,32,16,8), classes=3) # build UNet++
    # model = Unet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build U-Net
    # model = NestNet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build DLA

    model.compile(optimizer='Adam', loss='mean_squared_error')

    # train model
    
    modelCheckpoint=ModelCheckpoint(
        filepath=MODEL_WEIGHTS_DIR,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=1)

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[modelCheckpoint], validation_split=0.1)
    model.save_weights(MODEL_WEIGHTS_DIR)
    
    # Plot training & validation loss values
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    fig.savefig(os.path.join('data/loss.png'))
    """
    ##------------end of training-----------##
    
    ##---------------prediction-------------##    
    X_test, file_names = load_X(imgsz, X_TEST_DIR)

    input_channel_count = 3
    output_channel_count = 1
    first_layer_filter_count = 64
    model = Xnet(backbone_name='densenet121', encoder_weights=None, decoder_block_type='transpose',decoder_filters=(128,64,32,16,8), classes=3) # build UNet++
    model.load_weights(MODEL_WEIGHTS_DIR)
    BATCH_SIZE = 21
    Y_pred = model.predict(X_test, BATCH_SIZE)
    
    # Make directory
    if not os.path.exists(SAVE_PREDICT_DIR):
        os.makedirs(SAVE_PREDICT_DIR)
            
    for i, y in enumerate(Y_pred):
        img = cv2.imread(X_TEST_DIR + os.sep + file_names[i])
        y = cv2.resize(y, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(SAVE_PREDICT_DIR, file_names[i]), denormalize_y(y))
    ##-----------end of prediction--------##
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'data/unetpp', help='data dir')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=256, help='inference size h,w')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    opt = parser.parse_args()
    #print_args(FILE.stem, opt)
    return opt


def main(opt):
    #check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)